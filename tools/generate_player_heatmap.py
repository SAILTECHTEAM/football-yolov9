import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import cv2
import matplotlib
import json
import pandas as pd
import argparse
import os
from time import time
from scipy.ndimage import gaussian_filter

def load_tracking_data(filepath, jersey_number=None, team=None):
    # Initialize data structures
    tracks_df_data = {'track_id': [], 'frame': [], 'team': [], 'jersey_num': [], 
                     'x': [], 'y': [], 'team_conf': [], 'jersey_conf': []}
    track_metadata = {}
    # Parse the JSONL file
    with open(filepath, 'r') as f:
        for line in f:
            track = json.loads(line)
            if track['team'] == 'ball':
                continue
            if jersey_number is None and team is None:
                print("Please provide either jersey_number or team to filter the data.")
                return None, None
            if isinstance(track.get("jersey_num"), list):
                continue
            if jersey_number is not None and track['jersey_num'] != jersey_number:
                continue
            if team is not None and team not in track['team']:
                continue
            
            # Store metadata
            track_metadata[track['track_id']] = {
                'team': track['team'],
                'jersey_num': track['jersey_num'],
                'frame_range': track['frame_range'],
                'team_conf': track['team_conf'],
                'jersey_conf': track['jersey_conf']
            }
            
            # Store position data
            for i, frame in enumerate(track['frames']):
                tracks_df_data['track_id'].append(track['track_id'])
                tracks_df_data['frame'].append(frame)
                tracks_df_data['team'].append(track['team'])
                tracks_df_data['jersey_num'].append(track['jersey_num'])
                tracks_df_data['x'].append(track['projected'][i][0])
                tracks_df_data['y'].append(track['projected'][i][1])
                tracks_df_data['team_conf'].append(track['team_conf'])
                tracks_df_data['jersey_conf'].append(track['jersey_conf'])
        
    # Create DataFrame and set index
    if not tracks_df_data['track_id']:  # Check if data was found
        print(f"No data found for jersey_number={jersey_number}, team={team}")
        return pd.DataFrame(), {}
    
    # Create DataFrame and set index
    tracks_df = pd.DataFrame(tracks_df_data)
    
    return tracks_df, track_metadata

def fast_binned_heatmap(tracks_df, field_size=(1060, 660), bins=(100, 60)):
    """
    Generate a heatmap using numpy's histogram2d which is much faster than KDE
    
    Parameters:
    - tracks_df: DataFrame containing tracking data
    - field_size: dimensions of the football field
    - bins: number of bins for the histogram (resolution of the heatmap)
    
    Returns:
    - heatmap: 2D array containing binned counts with gaussian smoothing
    - xedges: bin edges along x-axis
    - yedges: bin edges along y-axis
    """
    # Use numpy's histogram2d which is very fast
    heatmap, xedges, yedges = np.histogram2d(
        tracks_df['x'].values, tracks_df['y'].values, 
        bins=bins, range=[[0, field_size[0]], [0, field_size[1]]]
    )
    # Apply smoothing if needed
    heatmap = gaussian_filter(heatmap, sigma=1.5)
    return heatmap, xedges, yedges

def generate_heatmap(tracks_df, field_size=(1060, 660), bg_img=None, cmap='hot', jersey_number=None, team=None):
    """
    Generate position heatmap for player(s) in the tracks_df
    
    Parameters:
    - tracks_df: DataFrame containing tracking data (already filtered for specific player if needed)
    - field_size: dimensions of the football field
    - bg_img: background image to overlay the heatmap on
    - cmap: colormap for the heatmap
    - jersey_number: jersey number (for title only)
    - team: team name (for title only)
    """

    if tracks_df.empty:
        print("No tracking data available for the specified player or team.")
        return None
    
    # Create heatmap using kernel density estimation
    x = tracks_df['x'].values
    y = tracks_df['y'].values

    # Set title based on filtering
    if jersey_number is not None and team is not None:
        title = f"Position Heatmap for Player #{jersey_number} ({team})"
    elif team is not None:
        title = f"Position Heatmap for Team {team}"
    else:
        title = "Position Heatmap"

    # Use fast binned heatmap instead of KDE
    bins = (int(field_size[0]/8), int(field_size[1]/8))
    heatmap, xedges, yedges = fast_binned_heatmap(tracks_df, field_size, bins)
    # # Create meshgrid
    # xi, yi = np.mgrid[0:field_size[0]:100j, 0:field_size[1]:50j]
    
    # # Compute kernel density
    # k = gaussian_kde([x, y])
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Add background image if provided
    if bg_img is not None:
        bg_img = cv2.resize(bg_img, (field_size[0], field_size[1]))
        bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        ax.imshow(bg_rgb, extent=[0, field_size[0], 0, field_size[1]])
    
    # Create a custom colormap with transparency at the low end
    base_cmap = matplotlib.colormaps[cmap]
    base_cmap.set_gamma(0.5)
    base_cmap.set_under("white")

    # # Create heatmap and store the mappable object
    # mesh = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=base_cmap, alpha=0.7)
    
    # Create heatmap using pcolormesh with the binned data
    mesh = ax.pcolormesh(xedges, yedges, heatmap.T, shading='auto', 
                       cmap=base_cmap, alpha=0.7)

    # Now use the mesh object for the colorbar
    # plt.colorbar(mesh, label='Density')
    
    ax.set_title(title)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Draw field boundaries
    ax.set_xlim(0, field_size[0])
    ax.set_ylim(0, field_size[1])
    
    plt.tight_layout()
    return fig

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate player position heatmap.")
    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to tracking data JSONL file.")
    parser.add_argument("--bg-img-path", type=str, default=None, help="Path to background image file.")
    parser.add_argument("--field-size", type=int, nargs=2, default=(1060, 660), help="Size of the football field (width height).")
    parser.add_argument("--jersey_number", type=int, default=None, help="Specific player jersey number to visualize.")
    parser.add_argument("--team", type=str, default=None, help="Team name to visualize ('home', 'away').")
    args = parser.parse_args()

    # Load tracking data
    jsonl_path = args.jsonl_path
    jersey_number = args.jersey_number
    team = args.team
    
    start_time = time()
    print("⏳ Loading tracking data...")
    # Load data for specific player if specified
    tracks_df, track_metadata = load_tracking_data(
        jsonl_path, 
        jersey_number=jersey_number if jersey_number else None, 
        team=team if team else None
    )
    middle_time = time()
    print(f"✅ Tracking data loaded in {middle_time - start_time:.2f} seconds.")
    if tracks_df.empty:
        print("⚠️ No data found. Please check jersey number and team values.")
        exit(1)
        
    # Load background image if provided
    bg_img = None
    if args.bg_img_path:
        bg_img = cv2.imread(args.bg_img_path)
        bg_img = cv2.resize(bg_img, (args.field_size[0], args.field_size[1]))
        if bg_img is None:
            print(f"⚠️ Warning: Could not load background image from {args.bg_img_path}")
    
    # Generate heatmap
    heatmap_fig = generate_heatmap(
        tracks_df,
        bg_img=bg_img,
        jersey_number=jersey_number,
        team=team
    )
    output_name = f"{team}_{jersey_number}" if jersey_number else f"{team}"
    output_path = os.path.dirname(jsonl_path)
    output_path = os.path.join(output_path, f"{output_name}_heatmap.png")

    # Save the heatmap
    if heatmap_fig:
        heatmap_fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Heatmap saved to {output_path}")
        end_time = time()
        print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
        # plt.show()
    else:
        print("❌ Failed to generate heatmap.")