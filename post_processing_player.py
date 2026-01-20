import argparse
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import compress
import json
import numpy as np
from collections import defaultdict
from scipy.signal import savgol_filter

import time
from typing import List, Dict, Any, Tuple, Iterator, Optional, Union, Set

from tools.remove_track_sharp import process_jsonl_detect_replace


# ================= Data Classes ================== #
@dataclass
class RawTrack:
    """
    Raw track data directly from object detection/tracking inference.
    Frame-level predictions before any aggregation.
    """
    track_id: str
    frames: List[int]
    projected: List[List[float]]  # [[x, y], ...]
    bbox: List[List[float]]  # [[x1, y1, x2, y2, conf], ...]
    team_conf: List[Dict[str, float]]  # [{"home": 0.8, "away": 0.2, ...}, ...] per frame
    jersey_num: List[int]  # [1, 10, 10, -1, ...] per frame (-1 = no detection)
    jersey_conf: List[List[float]]  # [[0.9], [0.95, 0.99], ...] per frame
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawTrack":
        """Load from JSONL dict."""
        return cls(
            track_id=data["track_id"],
            frames=data["frame_id"], # idk why it's named frame_id in jsonl
            projected=data["projected"],
            bbox=data.get("bbox", []),
            team_conf=data.get("team_conf", []),
            jersey_num=data.get("jersey_num", []),
            jersey_conf=data.get("jersey_conf", []),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_id": self.frames,
            "projected": self.projected,
            "bbox": self.bbox,
            "team_conf": self.team_conf,
            "jersey_num": self.jersey_num,
            "jersey_conf": self.jersey_conf,
        }


@dataclass
class TrackSegment:
    """
    Track segment after splitting by team changes.
    Team is now AGGREGATED (single value), but still frame-level jersey data.
    """
    track_id: str  # e.g., "123a", "123b"
    team: str  # "home" or "away" (voted)
    team_conf: float  # Aggregated confidence for this segment
    jersey_num: List[int]  # Still frame-level
    jersey_conf: List[List[float]]  # Still frame-level
    bbox_area: List[float]  # Derived from bbox
    frames: List[int]
    projected: List[List[float]]  # [[x, y], ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackSegment":
        """Load from JSONL dict."""
        return cls(
            track_id=data["track_id"],
            team=data["team"],
            team_conf=data.get("team_conf", 0.0),
            jersey_num=data.get("jersey_num", []),
            jersey_conf=data.get("jersey_conf", []),
            bbox_area=data.get("bbox_area", []),
            frames=data["frames"],
            projected=data["projected"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "team": self.team,
            "team_conf": self.team_conf,
            "jersey_num": self.jersey_num,
            "jersey_conf": self.jersey_conf,
            "bbox_area": self.bbox_area,
            "frames": self.frames,
            "projected": self.projected,
        }


@dataclass
class AggregatedTrack:
    """
    A track segment where the player identity (jersey number) has been 
    statistically analyzed.
    
    'jersey_num' now holds the result of the voting logic (single ID, list of candidates, or status string).
    """
    track_id: str
    team: str
    team_conf: float
    jersey_num: Union[str, int, List[int]]  # e.g. 10, [10, 7], "unsure", "NA"
    jersey_conf: Union[float, List[float]]  # e.g. 0.95, [0.8, 0.6]
    count: Union[int, List[int]]            # e.g. 15, [10, 5], 0
    bbox_area: List[float]
    frames: List[int]
    projected: List[List[float]]
    frame_range: List[int]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedTrack":
        """Load from JSONL dict."""
        return cls(
            track_id=data["track_id"],
            team=data["team"],
            team_conf=data.get("team_conf", 0.0),
            jersey_num=data.get("jersey_num", "unsure"),
            jersey_conf=data.get("jersey_conf", 0.0),
            count=data.get("count", 0),
            bbox_area=data.get("bbox_area", []),
            frames=data["frames"],
            projected=data["projected"],
            frame_range=data.get("frame_range", []),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "team": self.team,
            "jersey_num": self.jersey_num,
            "jersey_conf": self.jersey_conf,
            "count": self.count,
            "frame_range": self.frame_range,
            "frames": self.frames,
            "projected": self.projected,
            "bbox_area": self.bbox_area,
            "team_conf": self.team_conf,
        }


# ================== Post-Processing Utilities ================== #
def frame_to_time(frame: int, fps: float = 29.97, format_output: bool = True) -> str:
    """
    Convert frame index to time based on FPS.

    Args:
        frame (int): Frame index.
        fps (float): Frames per second. Default is 29.97.
        format_output (bool): If True, return formatted time (HH:MM:SS.ms), else return seconds.

    Returns:
        str or float: Formatted timestamp or raw seconds.
    """
    seconds = frame / fps
    if not format_output:
        return seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"  # includes milliseconds


def stream_jsonl_tracks(path: str) -> Iterator[Dict[str, Any]]:
    """Generator: Stream JSONL tracks one at a time."""
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    # 1. Print the error so you know data is missing
                    print(f"❌ Corrupt JSON at line {line_num} in {path}: {e}")
                    
                    # 2. Continue to the next line (don't crash the pipeline)
                    continue


def write_jsonl_stream(path: str, data_stream: Iterator[Dict[str, Any]]) -> int:
    """
    Consume iterator and write to JSONL.
    Returns count of records written.
    """
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in data_stream:
            f.write(json.dumps(record) + "\n")
            count += 1
    return count


# ================== load_and_split_tracks related ================== #
def calculate_bbox_area(bboxes: List[List[float]]) -> List[float]:
    """
    Calculate area of multiple bounding boxes (vectorized version).

    Args:
        bboxes: List of bounding boxes, each as [x1, y1, x2, y2]

    Returns:
        List of areas corresponding to each bounding box
    """
    if not bboxes:
        return []

    bboxes_array = np.array(bboxes)  # Shape: (N, 4)
    widths = np.maximum(0.0, bboxes_array[:, 2] - bboxes_array[:, 0])
    heights = np.maximum(0.0, bboxes_array[:, 3] - bboxes_array[:, 1])
    areas = (widths * heights).tolist()
    return areas


def assign_team_by_majority_vote(team_conf_list: List[Dict[str, float]]) -> str:
    """
    Assign team label based on majority vote from team confidence list.

    Args:
        team_conf_list: List of dictionaries with team confidence scores.
    
    Returns:
        Assigned team label.
    """

    team_count = defaultdict(float)
    for conf in team_conf_list:
        for k, v in conf.items():
            team_count[k] += v
    return max(team_count, key=team_count.get) if team_count else "ball"


def index_to_letter_suffix(idx: int) -> str:
    """Return 'a', 'b', ..., 'z', 'aa', 'ab', ... as suffix."""
    letters = []
    while True:
        letters.append(chr(97 + (idx % 26)))
        idx = idx // 26
        if idx == 0:
            break
        idx -= 1  # offset for 0-based index
    return "".join(reversed(letters))


def split_track_by_sliding_window(
    track: RawTrack, window_size: int = 20, threshold: float = 0.8
) -> List[TrackSegment]:
    """
    Splits a track when a new team dominates a sliding window.

    Args:
        track: Original track.
        window_size: Size of the sliding window.
        threshold: Ratio of frames in the window needed for a team to trigger a split.

    Returns:
        List of TrackSegment objects.
    """
    team_conf_list = track.team_conf

    # Get dominant team label for each frame
    dominant_team_list = [max(conf, key=conf.get) if conf else "ball" for conf in team_conf_list]

    segments = []
    buffer = []
    i = 0
    current_team = assign_team_by_majority_vote(team_conf_list)

    while i < len(dominant_team_list):
        if dominant_team_list[i] == current_team:
            buffer.append(i)
            i += 1
            continue

        # Only check if enough room for a full window
        if i + window_size <= len(dominant_team_list):
            window = dominant_team_list[i : i + window_size]
            counter = defaultdict(int)
            for t in window:
                counter[t] += 1
            dominant_in_window = max(counter, key=counter.get)
            ratio = counter[dominant_in_window] / window_size

            if dominant_in_window != current_team and ratio >= threshold:
                segment_conf_list = [team_conf_list[j] for j in buffer]
                if segment_conf_list:
                    team_score = sum(
                        conf.get(current_team, 0.0) for conf in segment_conf_list
                    ) / len(segment_conf_list)
                else:
                    team_score = 0.0

                segment = TrackSegment(
                    track_id=f"{track.track_id}{chr(97 + len(segments))}",
                    frames=[track.frames[j] for j in buffer],
                    projected=[track.projected[j] for j in buffer],
                    bbox_area=calculate_bbox_area([track.bbox[j] for j in buffer]) if track.bbox else [],
                    team_conf=team_score,
                    team=current_team,
                    jersey_num=[track.jersey_num[j] for j in buffer] if track.jersey_num else [],
                    jersey_conf=[track.jersey_conf[j] for j in buffer] if track.jersey_conf else [],
                )
                segments.append(segment)
                buffer = []
                current_team = dominant_in_window
                # move window forward
                continue

        buffer.append(i)
        i += 1

    # Final segment
    if buffer:
        segment_conf_list = [team_conf_list[j] for j in buffer]
        team_score = sum(conf.get(current_team, 0.0) for conf in segment_conf_list) / len(
            segment_conf_list
        )
        segment = TrackSegment(
            track_id=f"{track.track_id}{chr(97 + len(segments))}",
            frames=[track.frames[j] for j in buffer],
            projected=[track.projected[j] for j in buffer],
            bbox_area=calculate_bbox_area([track.bbox[j] for j in buffer]) if track.bbox else [],
            team_conf=team_score,
            team=current_team,
            jersey_num=[track.jersey_num[j] for j in buffer] if track.jersey_num else [],
            jersey_conf=[track.jersey_conf[j] for j in buffer] if track.jersey_conf else [],
        )
        segments.append(segment)

    return segments


def validate_track_length(track_data: RawTrack, min_track_length: int) -> bool:
    """Pure predicate: Check if track meets minimum length."""
    projected = track_data.projected
    if not projected:
        return False
    
    # Filter None values
    valid_points = [pt for pt in projected if pt is not None]
    return len(valid_points) >= min_track_length


def filter_valid_points(
    track: RawTrack,
    field_size: Tuple[int, int] = (1060, 660),
) -> RawTrack:
    """
    Pure function: Filters points outside field boundaries.
    
    Returns:
        track: Filtered track with only in-bounds points.
    """

    # 1. Validation
    if len(track.projected) == 0:
        return None

    # 2. Geometry Calculation
    w, h = field_size
    
    # This generates a list of booleans: [True, False, True, ...]
    in_bounds = [
        (0 <= p[0] <= w) and (0 <= p[1] <= h)
        for p in track.projected
    ]

    # 3. Check if we filtered everything out
    # 'any' is faster than sum() for lists because it stops at the first True
    if not any(in_bounds): 
        return None

    # 4. Helper for List Slicing (Solves the "Unbound Variable" & "Repetition" issues)
    def filter_list(data: list) -> list:
        # Only try to compress if data exists, otherwise return empty list
        return list(compress(data, in_bounds)) if data else []

    # 5. Constructor with Inlining
    return RawTrack(
        track_id=track.track_id,
        frames=filter_list(track.frames),
        projected=filter_list(track.projected),
        bbox=filter_list(track.bbox),
        team_conf=filter_list(track.team_conf),
        jersey_num=filter_list(track.jersey_num),
        jersey_conf=track.jersey_conf,
    )


def segment_track_stream(
    track: RawTrack,
    field_size: Tuple[int, int],
    min_track_length: int,
    window_size: int,
    threshold: float,
) -> Iterator[Dict[str, Any]]:
    """
    Generator: Process single track through full pipeline.
    Yields formatted track segments.
    """
    # Step 1: Validate minimum length (early exit)
    if not validate_track_length(track, min_track_length):
        return
    
    # Step 2: Filter None values  
    track.projected = [pt for pt in track.projected if pt is not None]
    
    # Step 3: Filter out-of-bounds points
    filtered_track = filter_valid_points(
        track,
        field_size=field_size,
    )

    if filtered_track is None:
        return
    
    if len(filtered_track.projected) < min_track_length:
        return
    
    # Step 4: Reconstruct track object for splitter

    # Step 5: Split track by team changes
    split_segments = split_track_by_sliding_window(
        filtered_track, window_size=window_size, threshold=threshold
    )
    
    # Step 6: Format each segment
    for segment in split_segments:
        if len(segment.projected) == 0:
            continue
        
        yield segment


def load_and_split_tracks(
    json_path: str,
    output_path: str,
    field_size: List[int],
    min_track_length: int,
    window_size: int,
    threshold: float,
):
    """
    Main entry point: Stream-process tracks with splitting and filtering.
    """
    # Create processing pipeline (lazy evaluation)
    input_stream = stream_jsonl_tracks(json_path)
    
    processed_stream = (
        segment
        for track_dict in input_stream
        for segment in segment_track_stream(
            track=RawTrack.from_dict(track_dict),
            field_size=tuple(field_size),
            min_track_length=min_track_length,
            window_size=window_size,
            threshold=threshold,
        )
    )
    
    # Execute and write
    count = write_jsonl_stream(output_path, (seg.to_dict() for seg in processed_stream))
    print(f"✅ Processed {count} track segments → {output_path}")


# # ================== determine_track_jersey_number related ================== #
@dataclass
class JerseyAnalysisResult:
    """
    Result of jersey number statistical analysis.
    
    Attributes:
        final_nums: List of jersey numbers sorted by (count, confidence)
        final_confs: Corresponding confidence scores
        counts: Detection counts for each jersey number
        status: "confirmed", "unsure", "NA", "ball"
    """
    final_nums: List[int] = field(default_factory=list)
    final_confs: List[float] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)
    status: str = "unsure"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-compatible dict for track output."""
        if self.status in ["NA", "unsure", "ball"]:
            return {
                "jersey_num": self.status,
                "jersey_conf": 0.0,
                "count": 0,
            }
        
        return {
            "jersey_num": self.final_nums,
            "jersey_conf": self.final_confs,
            "count": self.counts,
        }


def analyze_jersey_stats(
    team: str,
    raw_nums: List[int],
    raw_confs: List[List[float]],
    threshold: float,
    min_entries: int
) -> JerseyAnalysisResult:
    """
    Pure function: Calculates dominant jersey numbers from raw frame data.
    
    Args:
        team: Team label ("home", "away", "referee", "ball")
        raw_nums: Frame-level jersey number predictions
        raw_confs: Corresponding confidence scores per frame
        threshold: Minimum confidence to accept a prediction
        min_entries: Minimum detections needed to confirm a jersey number
    
    Returns:
        JerseyAnalysisResult with status and (optionally) confirmed numbers
    
    Examples:
        >>> analyze_jersey_stats("home", [10, 10, 7], [[0.99], [0.98], [0.85]], 0.95, 2)
        JerseyAnalysisResult(final_nums=[10], final_confs=[0.985], counts=[2], status="confirmed")
    """
    # 1. Early Exit: Special Teams
    if team == "referee":
        return JerseyAnalysisResult(status="NA")
    
    # Note: "ball" should be handled by caller (early exit before calling this function)
    
    # 2. Validation
    if not raw_nums or not raw_confs or len(raw_nums) != len(raw_confs):
        return JerseyAnalysisResult(status="unsure")
    
    # 3. Filtering Candidates (High-Confidence Only)
    candidates = []
    for num, conf_list in zip(raw_nums, raw_confs):
        # Skip invalid detections
        if num == -1 or not isinstance(conf_list, list) or not conf_list:
            continue
        
        # Strict threshold: ALL confidences in this frame must pass
        if any(c < threshold for c in conf_list):
            continue
        
        avg_conf = sum(conf_list) / len(conf_list)
        candidates.append({"num": num, "conf": avg_conf})
    
    if not candidates:
        return JerseyAnalysisResult(status="unsure")
    
    # 4. Aggregation (Group by Jersey Number)
    stats = defaultdict(list)
    for c in candidates:
        stats[c["num"]].append(c["conf"])
    
    # 5. Final Selection (Min Detection Count Filter)
    accepted = []
    for num, confs in stats.items():
        if len(confs) >= min_entries:
            accepted.append({
                "num": num,
                "mean_conf": sum(confs) / len(confs),
                "count": len(confs)
            })
    
    if not accepted:
        return JerseyAnalysisResult(status="unsure")
    
    # 6. Sorting (Count desc, then Confidence desc)
    accepted.sort(key=lambda x: (x["count"], x["mean_conf"]), reverse=True)
    
    return JerseyAnalysisResult(
        final_nums=[x["num"] for x in accepted],
        final_confs=[x["mean_conf"] for x in accepted],
        counts=[x["count"] for x in accepted],
        status="confirmed"
    )


def jersey_determination_stream(
    track: TrackSegment,
    confidence_threshold: float,
    min_accepted_entries: int,
) -> AggregatedTrack:
    """
    Pure function: Transforms a Raw Segment into an Aggregated Track.
    Handles ALL teams (Ball, Ref, Player) here.
    """
    
    # 1. Default / Fallback Values
    final_jersey_num = "unsure"
    final_jersey_conf = 0.0
    final_count = 0

    # 2. Logic Branching
    if track.team == "ball":
        # Balls are always unsure/0
        pass 
        
    elif track.team == "referee":
         final_jersey_num = "NA"
         
    else:
        # 3. Complex Analysis for Players
        stats = analyze_jersey_stats(
            team=track.team,
            raw_nums=track.jersey_num,
            raw_confs=track.jersey_conf,
            threshold=confidence_threshold,
            min_entries=min_accepted_entries
        )
        
        # Unpack the result object
        if stats.status == "confirmed":
            final_jersey_num = stats.final_nums
            final_jersey_conf = stats.final_confs
            final_count = stats.counts
        elif stats.status == "NA":
             final_jersey_num = "NA"
        # else: remains "unsure"

    # 4. Construct AggregatedTrack
    yield AggregatedTrack(
        track_id=track.track_id,
        team=track.team,
        team_conf=track.team_conf,
        jersey_num=final_jersey_num,
        jersey_conf=final_jersey_conf,
        count=final_count,
        bbox_area=track.bbox_area,
        frames=track.frames,
        projected=track.projected,
        frame_range=[min(track.frames), max(track.frames)] if track.frames else [],
    )


def determine_track_jersey_number(
    jsonl_path: str,
    output_path: str,
    confidence_threshold: float = 0.99,
    min_accepted_entries: int = 3,
):
    """
    Stream-process tracks to determine final jersey numbers.
    
    Args:
        jsonl_path: Input track JSONL (with frame-level jersey predictions)
        output_path: Output JSONL (with aggregated jersey numbers)
        confidence_threshold: Minimum confidence to accept a prediction (0-1)
        min_accepted_entries: Minimum detections needed to confirm a number
    """
    print(f"🔢 Determining jersey numbers: {jsonl_path} → {output_path}")
    
    # Create processing pipeline (lazy evaluation)
    input_stream = stream_jsonl_tracks(jsonl_path)
    
    processed_stream = (
        track for track_dict in input_stream
        for track in jersey_determination_stream(
            track=TrackSegment.from_dict(track_dict),
            confidence_threshold=confidence_threshold,
            min_accepted_entries=min_accepted_entries,
        )
    )

    # Execute and write
    count = write_jsonl_stream(
        output_path,
        (track.to_dict() for track in processed_stream)  # ← Dict conversion here
    )



# ================== hybrid_merge_stream_fixed related ================== #
@dataclass
class ActiveTrack:
    """
    Represents a track currently being merged in the buffer.
    Encapsulates the logic of 'adding a segment'.
    """
    track_id: str
    team: str
    frames: List[int]
    projected: List[List[float]]
    bbox_area: List[float]
    
    # Jersey Accumulators
    jersey_entries: List[Any] = field(default_factory=list) # Stores {'num':..., 'conf':..., 'count':...}
    
    # Confidence Accumulators
    team_conf_sum: float = 0.0
    team_conf_count: int = 0
    
    @classmethod
    def from_segment(cls, seg: Dict[str, Any]) -> 'ActiveTrack':
        """Initialize from the first segment."""
        # Helper to normalize jersey data to a consistent list format
        j_nums = seg.get("jersey_num", [])
        j_confs = seg.get("jersey_conf", [])
        j_counts = seg.get("count", 0)
        
        # Handle scalar/list mismatch normalization here...
        # (Simplified for brevity, assuming normalized lists)
        entries = []
        if isinstance(j_nums, list):
            for n, c, cnt in zip(j_nums, j_confs, j_counts if isinstance(j_counts, list) else [j_counts]*len(j_nums)):
                entries.append({'num': n, 'conf': c, 'count': cnt})
        elif j_nums not in ["unsure", "NA"]:
             entries.append({'num': j_nums, 'conf': j_confs, 'count': j_counts})

        return cls(
            track_id=seg["track_id"],
            team=seg["team"],
            frames=seg["frames"],
            projected=seg["projected"],
            bbox_area=seg.get("bbox_area", []),
            jersey_entries=entries,
            team_conf_sum=seg.get("team_conf", 0.0) * len(seg["frames"]),
            team_conf_count=len(seg["frames"])
        )

    def can_merge(self, seg: Dict[str, Any], max_gap: int, max_dist: float, max_overlap: int) -> bool:
        """Pure Predicate: Should we merge this segment?"""
        if seg["team"] != self.team:
            return False
            
        last_frame = self.frames[-1]
        start_frame = seg["frames"][0]
        gap = start_frame - last_frame

        # 1. Temporal Check
        if not ((0 <= gap <= max_gap) or (0 < -gap <= max_overlap)):
            return False

        # 2. Spatial Check
        # Use numpy for fast distance
        p1 = np.array(self.projected[-1])
        p2 = np.array(seg["projected"][0])
        dist = np.linalg.norm(p1 - p2)
        
        return dist <= max_dist

    def merge(self, seg: Dict[str, Any]):
        """Mutator: Absorb the new segment data."""
        # 1. Merge Spatiotemporal Data
        self.frames.extend(seg["frames"])
        self.projected.extend(seg["projected"])
        self.bbox_area.extend(seg.get("bbox_area", []))
        
        # 2. Deduplicate frames (keep first occurrence for each frame)
        frame_dict = {}
        for f, p, a in zip(self.frames, self.projected, self.bbox_area):
            if f not in frame_dict:
                frame_dict[f] = (p, a)
        
        # Sort by frame and reconstruct
        sorted_frames = sorted(frame_dict.keys())
        self.frames = sorted_frames
        self.projected = [frame_dict[f][0] for f in sorted_frames]
        self.bbox_area = [frame_dict[f][1] for f in sorted_frames]

        # 3. Merge Confidence
        self.team_conf_sum += seg.get("team_conf", 0.0) * len(seg["frames"])
        self.team_conf_count += len(seg["frames"])

        # 4. Merge Jersey Data - Extract and append entries
        new_entries = _extract_jersey_entries(seg)
        self.jersey_entries.extend(new_entries)


def _extract_jersey_entries(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract jersey entries from a segment, handling all formats.
    
    Returns:
        List of dicts with keys: 'num', 'conf', 'count'
    """
    j_nums = seg.get("jersey_num", "unsure")
    j_confs = seg.get("jersey_conf", 0.0)
    j_counts = seg.get("count", 0)
    
    # Handle special statuses
    if j_nums in ["unsure", "NA"]:
        return []
    
    # Normalize to lists
    if not isinstance(j_nums, list):
        j_nums = [j_nums]
    if not isinstance(j_confs, list):
        j_confs = [j_confs]
    if not isinstance(j_counts, list):
        j_counts = [j_counts]
    
    # Zip together (handle length mismatches)
    entries = []
    for i in range(len(j_nums)):
        entries.append({
            'num': j_nums[i],
            'conf': j_confs[i] if i < len(j_confs) else 0.0,
            'count': j_counts[i] if i < len(j_counts) else 0
        })
    
    return entries


def finalize_track_data(
    track: ActiveTrack, 
    smoothing_window: int, 
    polyorder: int
) -> AggregatedTrack:
    """
    Takes an ActiveTrack, performs interpolation/smoothing/jersey-voting, 
    and returns an AggregatedTrack.
    """
    # 1. Interpolation & Smoothing
    # (Assuming interpolate_full_track is defined elsewhere)
    frames_arr = np.array(track.frames)
    projected_arr = np.array(track.projected)
    areas_arr = np.array(track.bbox_area)
    
    # Call your existing helper
    frames, projected, areas = interpolate_full_track(frames_arr, projected_arr, areas_arr)
    if len(projected) >= smoothing_window:
        xs = savgol_filter(projected[:, 0], smoothing_window, polyorder)
        ys = savgol_filter(projected[:, 1], smoothing_window, polyorder)
        projected = np.stack([xs, ys], axis=1)
    # 2. Resolve Jersey Number (vote across all merged segments)
    final_jersey, final_conf, final_count = _resolve_jersey_vote(track)

    # 3. Calculate Team Confidence
    avg_team_conf = (
        track.team_conf_sum / track.team_conf_count 
        if track.team_conf_count > 0 
        else 0.0
    )
    
    # 4. Return AggregatedTrack
    return AggregatedTrack(
        track_id=track.track_id,
        team=track.team,
        team_conf=avg_team_conf,
        jersey_num=final_jersey,
        jersey_conf=final_conf,
        count=final_count,
        frames=frames.tolist(),
        projected=projected.tolist(),
        bbox_area=areas.tolist(),
        frame_range=[int(frames[0]), int(frames[-1])] if len(frames) > 0 else [0, 0],
    )


def _resolve_jersey_vote(track: ActiveTrack) -> Tuple[Union[str, int, List[int]], Union[float, List[float]], Union[int, List[int]]]:
    """
    Consolidate jersey entries and determine final jersey number(s).
    
    This implements the merge logic from the original hybrid_merge_stream_fixed():
    1. Referee → "NA"
    2. No entries → "unsure"
    3. Multiple entries → Aggregate by number, sort by (count, conf)
    
    Returns:
        (jersey_num, jersey_conf, count) tuple
        - jersey_num: "NA", "unsure", int, or List[int]
        - jersey_conf: float or List[float]
        - count: int or List[int]
    """
    # CASE 1: Referee team - always "NA"
    if track.team == "referee":
        return "NA", 0.0, 0
    
    # CASE 2: No jersey entries collected
    if not track.jersey_entries:
        return "unsure", 0.0, 0
    
    # CASE 3: Aggregate entries by jersey number
    # Create mapping: jersey_num -> {"confs": [...], "counts": [...]}
    jersey_data_map = defaultdict(lambda: {"confs": [], "counts": []})
    
    for entry in track.jersey_entries:
        num = entry["num"]
        conf = entry["conf"]
        count = entry["count"]
        
        jersey_data_map[num]["confs"].append(conf)
        jersey_data_map[num]["counts"].append(count)
    
    # Average confidences and sum counts for each jersey number
    merged_jerseys = []
    merged_confs = []
    merged_counts = []
    
    for jersey_num in sorted(jersey_data_map.keys()):
        conf_list = jersey_data_map[jersey_num]["confs"]
        count_list = jersey_data_map[jersey_num]["counts"]
        
        # Average confidence (weighted by count if you prefer)
        avg_conf = sum(conf_list) / len(conf_list)
        total_count = sum(count_list)
        
        merged_jerseys.append(jersey_num)
        merged_confs.append(avg_conf)
        merged_counts.append(total_count)
    
    # Sort by count (descending), then by confidence (descending)
    sorted_triplets = sorted(
        zip(merged_jerseys, merged_confs, merged_counts),
        key=lambda x: (x[2], x[1]),  # (count, conf)
        reverse=True
    )
    
    # Extract sorted lists
    final_nums = [num for num, _, _ in sorted_triplets]
    final_confs = [conf for _, conf, _ in sorted_triplets]
    final_counts = [count for _, _, count in sorted_triplets]
    
    # Return format depends on number of candidates
    if len(final_nums) == 1:
        # Single jersey number - return as singletons
        return [final_nums[0]], [final_confs[0]], [final_counts[0]]
    else:
        # Multiple candidates - return as lists
        return final_nums, final_confs, final_counts


def interpolate_full_track(
    frames: List[int], points: np.ndarray, areas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate full track to fill in all missing frames using linear interpolation.

    Args:
        frames (List[int]): List of frame indices.
        points (np.ndarray): Corresponding points (N, 2) for each frame.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Interpolated frames and points (in same order).
    """
    if len(frames) < 2:
        return np.array(frames), points, areas  # Not enough points to interpolate

    all_frames = np.arange(frames[0], frames[-1] + 1)
    xs_interp = np.interp(all_frames, frames, points[:, 0])
    ys_interp = np.interp(all_frames, frames, points[:, 1])
    full_points = np.stack([xs_interp, ys_interp], axis=1)
    if areas is None or len(areas) == 0:
        areas_interp = np.zeros_like(all_frames, dtype=float)
    else:
        areas_interp = np.interp(all_frames, frames, areas)

    return all_frames, full_points, areas_interp


def hybrid_merge_stream_fixed(
    jsonl_path: str,
    output_path: str,
    max_merge_gap: int = 5,
    max_merge_overlap_frames: int = 3,
    max_merge_distance: float = 10,
    smoothing_window: int = 11,
    polyorder: int = 3,
):
    """
    Stream-based track merging with object-oriented state management.
    
    Args:
        jsonl_path: Input JSONL with track segments
        output_path: Output JSONL with merged tracks
        max_merge_gap: Maximum frame gap for merging
        max_merge_overlap_frames: Maximum frame overlap allowed
        max_merge_distance: Maximum spatial distance for merging
        smoothing_window: Window size for Savitzky-Golay filter
        polyorder: Polynomial order for smoothing
    """
    # 1. Load Data (Still loaded fully, but isolated)
    # If optimization is needed later, this is the ONLY part to change.
    frame_to_tracks = _load_segments_by_frame(jsonl_path)
    max_buffer_frame = max(frame_to_tracks.keys()) if frame_to_tracks else 0
    
    active_tracks: Dict[str, ActiveTrack] = {} # Map ID -> Object
    done_tracks: Set[str] = set()
    
    with open(output_path, "w") as f_out:
        
        # 2. Temporal Loop
        for current_frame in range(max_buffer_frame + 1):
            
            # A. Gather Candidates
            candidates = []
            for offset in range(-max_merge_overlap_frames, max_merge_gap + 1):
                candidates.extend(frame_to_tracks.get(current_frame + offset, []))

            # B. Match Logic
            merged_this_round = set()
            
            for seg in candidates:
                tid = seg["track_id"]
                if tid in done_tracks or tid in merged_this_round:
                    continue

                if tid in active_tracks:
                    continue

                # Find Best Match in Active Tracks
                best_match_id = None
                best_dist = float("inf")

                for mtid, active_track in active_tracks.items():
                    # Use the method on the class!
                    if active_track.can_merge(seg, max_merge_gap, max_merge_distance, max_merge_overlap_frames):
                        # Calculate specific distance for tie-breaking
                        p1 = np.array(active_track.projected[-1])
                        p2 = np.array(seg["projected"][0])
                        dist = np.linalg.norm(p1 - p2)
                        
                        if dist < best_dist:
                            best_dist = dist
                            best_match_id = mtid

                # Perform Merge or Create New
                if best_match_id:
                    active_tracks[best_match_id].merge(seg)
                    merged_this_round.add(tid)
                    done_tracks.add(tid)
                else:
                    # Create new object
                    new_track = ActiveTrack.from_segment(seg)
                    active_tracks[tid] = new_track
                    merged_this_round.add(tid)

            # C. Flush Stale Tracks
            # Collect IDs first to avoid "dictionary changed size during iteration"
            stale_ids = [
                tid for tid, track in active_tracks.items()
                if track.frames[-1] < current_frame - max_merge_gap
            ]

            for tid in stale_ids:
                track = active_tracks.pop(tid)
                # Call helper to smooth/write
                final_track: AggregatedTrack = finalize_track_data(
                    track, 
                    smoothing_window, 
                    polyorder
                )
                f_out.write(json.dumps(final_track.to_dict()) + "\n")
                done_tracks.add(tid) # Just in case

        # 3. Final Flush (Whatever is left in buffer)
        for track in active_tracks.values():
            final_track: AggregatedTrack = finalize_track_data(track, smoothing_window, polyorder)
            f_out.write(json.dumps(final_track.to_dict()) + "\n")

    print(f"✅ Merged and saved to: {output_path}")


def _load_segments_by_frame(path: str) -> Dict[int, List[dict]]:
    """
    Load segments indexed by their start frame.
    Isolated loader that can be swapped for a buffered generator later.
    """
    mapping = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    seg = json.loads(line)
                    start_frame = seg["frames"][0]
                    mapping[start_frame].append(seg)
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"⚠️  Skipping invalid segment: {e}")
                    continue
    return mapping


# ================== Endpoint Anomaly Detection ================== #
def trim_single_track(
    track: AggregatedTrack,  # <--- Reusing your existing class
    # Explicit parameters
    window_size: int,
    speed_threshold_factor: float,
    acceleration_threshold_factor: float,
    direction_change_threshold: float,
    check_start: bool,
    check_end: bool,
    min_track_length: int
) -> Optional[AggregatedTrack]:
    

    # 1. Filter: Skip Ball
    if track.team in ["ball"]:
        return track

    # 2. Filter: Too short
    if len(track.frames) < window_size * 2 or len(track.frames) < min_track_length:
        return track

    # 3. Detect (Pass args through)
    try:
        start_idx, end_idx = detect_endpoint_anomalies(
            track.projected, track.frames,
            window_size, speed_threshold_factor, 
            acceleration_threshold_factor, direction_change_threshold,
            check_start, check_end
        )
    except Exception as e:
        print(f"⚠️  Error processing track {track.track_id}: {e}")
        return track

    # 4. Validate Trim
    new_len = end_idx - start_idx + 1
    if start_idx >= end_idx or new_len < min_track_length:
        return None

    if start_idx == 0 and end_idx == len(track.frames) - 1:
        return track
    
    # Helper to slice lists safely
    def slice_safe(lst):
        return lst[start_idx : end_idx + 1] if lst and len(lst) == len(track.frames) else lst

    return replace(
        track,
        frames=track.frames[start_idx : end_idx + 1],
        projected=track.projected[start_idx : end_idx + 1],
        bbox_area=slice_safe(track.bbox_area),
        frame_range=[int(track.frames[start_idx]), int(track.frames[end_idx])],
        # Note: If AggregatedTrack has jersey_conf as list, slice it too:
        # jersey_conf=slice_safe(track.jersey_conf)
    )


def trim_track_endpoints_streaming(
    jsonl_path: str,
    output_path: str,
    # Explicit arguments in the main entry point
    window_size: int = 61,
    speed_threshold_factor: float = 2.5,
    acceleration_threshold_factor: float = 2.5,
    direction_change_threshold: float = 60.0,
    check_start: bool = True,
    check_end: bool = True,
    min_track_length: int = 30,
):
    print(f"✂️ Trimming endpoints: {jsonl_path} → {output_path}")

    # 1. Prepare the Processor
    # We bake the configuration into a callable function
    processor_fn = partial(
        trim_single_track,
        window_size=window_size,
        speed_threshold_factor=speed_threshold_factor,
        acceleration_threshold_factor=acceleration_threshold_factor,
        direction_change_threshold=direction_change_threshold,
        check_start=check_start,
        check_end=check_end,
        min_track_length=min_track_length
    )

    # 2. Pipeline
    input_stream = stream_jsonl_tracks(jsonl_path)
    
    # 3. Execution
    # Note: AggregatedTrack needs a .from_dict() method as discussed previously
    processed_stream = (
        processor_fn(track=AggregatedTrack.from_dict(d))
        for d in input_stream
    )

    # 4. Sink
    # Filter out None (removed tracks) and handle parsing errors if from_dict returns None
    valid_stream = (t.to_dict() for t in processed_stream if t is not None)
    
    count = write_jsonl_stream(output_path, valid_stream)


def detect_endpoint_anomalies(positions: np.ndarray, frames: np.ndarray, 
                              window_size: int = 15,
                              speed_threshold_factor: float = 3.0,
                              acceleration_threshold_factor: float = 3.0,
                              direction_change_threshold: float = 60,
                              check_start: bool = True,
                              check_end: bool = True) -> Tuple[int, int]:
    """
    Detect anomalous movements at the start or end of a track.

    Args:
        positions (np.ndarray): Array of shape (N, 2) with x, y projected positions.
        frames (np.ndarray): Array of shape (N,) with frame indices.
        window_size (int): Number of frames to consider at start/end.
        speed_threshold_factor (float): Multiplier for speed threshold.
        acceleration_threshold_factor (float): Multiplier for acceleration threshold.
        direction_change_threshold (float): Angle in degrees for direction change.
        check_start (bool): Whether to check the start of the track.
        check_end (bool): Whether to check the end of the track.

    Returns:
        Tuple[int, int]: Indices to trim the start and end of the track.
    """
    n = len(positions)
    
    if n < window_size * 2:
        return 0, n - 1
    
    # Compute velocities and speeds
    dt = np.diff(frames)
    dt[dt == 0] = 1
    
    velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Check for valid speeds    
    if len(speeds) == 0:
        return 0, n - 1
    
    # Compute accelerations
    accelerations = np.diff(velocities, axis=0) / dt[:-1, np.newaxis]
    acc_magnitudes = np.linalg.norm(accelerations, axis=1)
    
    # Compute global statistics (excluding endpoints)
    mid_start = window_size
    mid_end = n - window_size
    
    if mid_end <= mid_start:
        mid_speeds = speeds
        mid_accs = acc_magnitudes
    else:
        mid_speeds = speeds[mid_start:mid_end]
        mid_accs = acc_magnitudes[mid_start:mid_end-1]

    # Check for empty mid_speeds and mid_accs before computing statistics
    if len(mid_speeds) == 0:
        # Use full speeds if mid section is empty
        mid_speeds = speeds
    if len(mid_accs) == 0:
        # Use full accelerations if mid section is empty
        mid_accs = acc_magnitudes
    
    # Final check - if still empty, return no trimming
    if len(mid_speeds) == 0:
        return 0, n - 1
    
    mean_speed = np.mean(mid_speeds)
    std_speed = np.std(mid_speeds)

    # Handle case where std_speed is 0 or NaN
    if np.isnan(std_speed) or std_speed == 0:
        std_speed = 1.0  # Default value to avoid division issues
    
    if len(mid_accs) > 0:
        mean_acc = np.mean(mid_accs)
        std_acc = np.std(mid_accs)
        if np.isnan(std_acc) or std_acc == 0:
            std_acc = 1.0
    else:
        mean_acc = 0.0
        std_acc = 1.0
    
    speed_threshold = mean_speed + speed_threshold_factor * std_speed
    acc_threshold = mean_acc + acceleration_threshold_factor * std_acc
    
    # Check start
    start_trim_idx = 0
    if check_start:
        for i in range(min(window_size, len(speeds))):
            if speeds[i] > speed_threshold:
                start_trim_idx = i + 1
                continue
            
            if i < len(acc_magnitudes) and acc_magnitudes[i] > acc_threshold:
                start_trim_idx = i + 1
                continue
            
            if i + 5 < len(velocities):
                current_dir = velocities[i]
                future_dir = velocities[i + 5]
                
                norm_current = np.linalg.norm(current_dir)
                norm_future = np.linalg.norm(future_dir)
                
                # Check for zero vectors
                if norm_current > 1e-6 and norm_future > 1e-6:
                    cos_angle = np.dot(current_dir, future_dir) / (norm_current * norm_future)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    
                    if angle_deg > direction_change_threshold:
                        start_trim_idx = i + 1
                        continue
            break
    
    # Check end
    end_trim_idx = n - 1
    if check_end:
        for i in range(min(window_size, len(speeds))):
            idx = len(speeds) - 1 - i
            
            if speeds[idx] > speed_threshold:
                end_trim_idx = idx
                continue
            
            if idx - 1 >= 0 and idx - 1 < len(acc_magnitudes):
                if acc_magnitudes[idx - 1] > acc_threshold:
                    end_trim_idx = idx
                    continue
            
            if idx - 5 >= 0:
                current_dir = velocities[idx]
                past_dir = velocities[idx - 5]
                
                norm_current = np.linalg.norm(current_dir)
                norm_past = np.linalg.norm(past_dir)
                
                # Check for zero vectors
                if norm_current > 1e-6 and norm_past > 1e-6:
                    cos_angle = np.dot(current_dir, past_dir) / (norm_current * norm_past)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    
                    if angle_deg > direction_change_threshold:
                        end_trim_idx = idx
                        continue
            break
    
    return start_trim_idx, end_trim_idx


def remove_tracks_near_boundary_stream(
    jsonl_path, output_jsonl_path, field_size, margin_meter=30, near_ratio_threshold=0.9
):
    """
    Removes tracks that stay near the field boundary for most of the time.

    Args:
        jsonl_path (str): Input path to .jsonl file.
        output_jsonl_path (str): Output path to write filtered tracks.
        field_size (tuple): Field dimensions (length, width) in 0.1 meters.
        margin_meter (float): Distance from edge considered "near".
        near_ratio_threshold (float): Ratio of points near edge to consider it a boundary-only track.
    """
    with open(jsonl_path, "r") as f_in, open(output_jsonl_path, "w") as f_out:
        for line in f_in:
            track = json.loads(line)
            team = track.get("team", "")
            points = np.array(track.get("projected", []))

            if len(points) == 0:
                continue  # skip empty tracks

            if team == "ball":
                f_out.write(json.dumps(track) + "\n")
                continue  # Always keep ball

            xs, ys = points[:, 0], points[:, 1]
            near_left = xs < margin_meter
            near_right = xs > (field_size[0] - margin_meter)
            near_top = ys < margin_meter
            near_bottom = ys > (field_size[1] - margin_meter)

            near_edge_mask = near_left | near_right | near_top | near_bottom
            near_edge_ratio = near_edge_mask.sum() / len(points)

            if near_edge_ratio < near_ratio_threshold:
                f_out.write(json.dumps(track) + "\n")


def remove_static_tracks(
    jsonl_path,
    output_jsonl_path,
    movement_threshold=20,  # in meters (10 = 1m if 0.1m units)
):
    """
    Remove tracks that don't move significantly.

    Args:
        jsonl_path (str): Input path to .jsonl file.
        output_jsonl_path (str): Output path to write filtered tracks.
        movement_threshold (float): Minimum total movement (Euclidean) to keep.
    """
    with open(jsonl_path, "r") as f_in, open(output_jsonl_path, "w") as f_out:
        for line in f_in:
            track = json.loads(line)
            # if track.get("team") != "ball":
            #     f_out.write(json.dumps(track) + "\n")
            #     continue

            points = np.array(track.get("projected", []))
            if len(points) < 2:
                continue  # skip too short

            # Compute total movement
            deltas = np.diff(points, axis=0)
            distances = np.linalg.norm(deltas, axis=1)
            total_distance = distances.sum()

            if total_distance >= movement_threshold:
                f_out.write(json.dumps(track) + "\n")


def coarse_postprocessing(
    json_path: str,
    home_jersey_numbers: List[int],
    away_jersey_numbers: List[int],
    field_size: List[int],
    min_track_length: int,
    smoothing_window: int,
    polyorder: int,
    max_merge_gap: int,
    max_merge_overlap_frames: int,
    max_merge_distance: int,
    window_size: int,
    threshold: float,
    endpoint_window_size: int = 61,
    endpoint_speed_factor: float = 2.5,
    endpoint_acceleration_factor: float = 2.5,
    endpoint_direction_threshold: float = 60,
    confidence_threshold: float = 0.99,
    min_accepted_entries: int = 7,
    detector_kwargs=None,
):

    start_load = time.time()
    # Merge and filter tracks
    load_and_split_tracks(
        json_path=json_path,
        output_path=json_path.replace(".jsonl", "_split.jsonl"),
        field_size=field_size,
        min_track_length=min_track_length,
        window_size=window_size,
        threshold=threshold,
    )
    end_load = time.time()
    print(f"✅ Loaded and split tracks in {end_load - start_load:.2f} seconds")

    determine_track_jersey_number(
        jsonl_path=json_path.replace(".jsonl", "_split.jsonl"),
        output_path=json_path.replace(".jsonl", "_split_with_jersey.jsonl"),
        confidence_threshold=confidence_threshold,
        min_accepted_entries=min_accepted_entries,
    )
    end_jersey = time.time()
    print(f"✅ Determined jersey numbers in {end_jersey - end_load:.2f} seconds")

    hybrid_merge_stream_fixed(
        jsonl_path=json_path.replace(".jsonl", "_split_with_jersey.jsonl"),
        output_path=json_path.replace(".jsonl", "_merged.jsonl"),
        max_merge_gap=max_merge_gap,
        max_merge_overlap_frames=max_merge_overlap_frames,
        max_merge_distance=max_merge_distance,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
    )
    end_merge = time.time()
    print(f"✅ Merged tracks in {end_merge - end_jersey:.2f} seconds")

    trim_track_endpoints_streaming(
        jsonl_path=json_path.replace(".jsonl", "_merged.jsonl"),
        output_path=json_path.replace(".jsonl", "_merged_trimmed.jsonl"),
        window_size=endpoint_window_size,
        speed_threshold_factor=endpoint_speed_factor,
        acceleration_threshold_factor=endpoint_acceleration_factor,
        direction_change_threshold=endpoint_direction_threshold,
        check_start=True,
        check_end=True,
        min_track_length=min_track_length,
    )
    end_trim = time.time()
    print(f"✅ Trimmed endpoint anomalies in {end_trim - end_merge:.2f} seconds")

    process_jsonl_detect_replace(
        input_path=json_path.replace(".jsonl", "_merged_trimmed.jsonl"),
        output_path=json_path.replace(".jsonl", "_merged_trimmed_detected.jsonl"),
        detector_kwargs=detector_kwargs,
    )


    remove_tracks_near_boundary_stream(
        jsonl_path=json_path.replace(".jsonl", "_merged_trimmed_detected.jsonl"),
        output_jsonl_path=json_path.replace(".jsonl", "_merged_filtered_near_boundary.jsonl"),
        field_size=field_size,
        margin_meter=30,
    )
    end_boundary = time.time()
    print(f"✅ Removed boundary-only tracks in {end_boundary - end_trim:.2f} seconds")

    remove_static_tracks(
        json_path.replace(".jsonl", "_merged_filtered_near_boundary.jsonl"),
        json_path.replace(".jsonl", "_merged_filtered.jsonl"),
        movement_threshold=30,  # in meters (10 = 1m if 0.1m units)
    )
    end_static_ball = time.time()
    print(f"✅ Removed static ball tracks in {end_static_ball - end_boundary:.2f} seconds")

    return json_path.replace(".jsonl", "_merged_filtered.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Process merged tracks from tracking JSONL")
    parser.add_argument(
        "--json-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the merged tracking JSONL files from different cameras",
    )
    parser.add_argument("--image-path", type=str, required=True, help="Path to the field image")
    parser.add_argument(
        "--home-jersey-numbers",
        type=int,
        nargs="+",
        required=True,
        help="List of home team jersey numbers",
    )
    parser.add_argument(
        "--away-jersey-numbers",
        type=int,
        nargs="+",
        required=True,
        help="List of away team jersey numbers",
    )
    parser.add_argument(
        "--field-size",
        type=int,
        nargs=2,
        default=[1060, 660],
        help="Field size (length, width) as 0.1m",
    )
    parser.add_argument(
        "--min-track-length", type=int, default=10, help="Minimum track length to keep"
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=90,
        help="Savitzky-Golay filter window size",
    )
    parser.add_argument("--polyorder", type=int, default=7, help="Polynomial order for smoothing")
    parser.add_argument(
        "--max-merge-gap",
        type=int,
        default=20,
        help="Max allowed gap (frames) between mergeable tracks",
    )
    parser.add_argument(
        "--max-merge-overlap-frames",
        type=int,
        default=15,
        help="Max allowed overlap for merging",
    )
    parser.add_argument(
        "--max-merge-distance",
        type=int,
        default=50,
        help="Max spatial distance for merging",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Window size for velocity consistency check",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for velocity consistency",
    )
    parser.add_argument(
        "--endpoint-window-size",
        type=int,
        default=61,
        help="Window size for endpoint checking (frames, default: 61 ≈ 2s @ 30fps)"
    )
    parser.add_argument(
        "--endpoint-speed-factor",
        type=float,
        default=2.5,
        help="Speed threshold factor for endpoint anomalies (default: 2.5 std)"
    )
    parser.add_argument(
        "--endpoint-acceleration-factor",
        type=float,
        default=2.5,
        help="Acceleration threshold factor (default: 2.5 std)"
    )
    parser.add_argument(
        "--endpoint-direction-threshold",
        type=float,
        default=60,
        help="Direction change threshold in degrees (default: 60°)"
    )   
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    # second pass with tighter gates
    DETECTOR = dict(
        window_size=501,
        step=250,
        prominence=2,
        min_wave_len=0,
        max_wave_len=60,
        speed_std_factor=None,
        smooth_window=0,
        savgol_poly=2,
        min_steepness=0.1,
        min_quad_curv=0.9,
        min_monotonic_ratio=0.5,
        max_gap_size=30,
    )
    
    start = time.time()

    # # Stage 1: Process each camera
    processed_files = []
    for json_path in args.json_paths:
        print(f"Processing camera JSONL: {json_path}")
        output = coarse_postprocessing(
            json_path=json_path,
            home_jersey_numbers=args.home_jersey_numbers,
            away_jersey_numbers=args.away_jersey_numbers,
            field_size=tuple(args.field_size),
            min_track_length=args.min_track_length,
            smoothing_window=args.smoothing_window,
            polyorder=args.polyorder,
            max_merge_gap=args.max_merge_gap,
            max_merge_overlap_frames=args.max_merge_overlap_frames,
            max_merge_distance=args.max_merge_distance,
            window_size=args.window_size,
            threshold=args.threshold,
            endpoint_window_size=args.endpoint_window_size,
            endpoint_speed_factor=args.endpoint_speed_factor,
            endpoint_acceleration_factor=args.endpoint_acceleration_factor,
            endpoint_direction_threshold=args.endpoint_direction_threshold,
            detector_kwargs=DETECTOR,
        )
        processed_files.append(output)

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()

# example usage:
# python3 post_processing_player.py \
#  --json-paths "./runs/detect/test_4k_player_640/team_tracking.jsonl" \
#  --image-path "./data/images/mongkok_football_field.png" \
#  --home-jersey-numbers 1 2 3 4 7 10 11 16 20 27 30 13 23 25 8 14 17 18 21 24 31 33 34 \
#  --away-jersey-numbers 26 2 6 7 9 16 20 30 36 77 99 1 17 22 23 24 28 33 42 43 44 72 88 \
