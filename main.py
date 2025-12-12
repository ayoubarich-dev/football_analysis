from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from tqdm import tqdm
import numpy as np

def main():
    #read video
    print("🎬 Reading video...")
    video_frames = read_video("input_videos/08fd33_4.mp4")
    
    #initialize tracker
    print("🤖 Initializing tracker and getting object tracks...")
    
    tracker = Tracker("models/best.pt")
    
    print("⚡ Using stub for tracks...")
    
    tracks = tracker.get_object_trackes(video_frames,read_from_stub=True,stub_path="stubs/track_stubs.pkl")
    
    # Interpolate Ball Positions
    print("🧮 Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    #Assign team colors
    print("🎨 Assigning team colors...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks["players"][0]
                                    )
    for frame_num, player_track in enumerate(tqdm(tracks["players"], desc="Assigning team colors to players")):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tqdm(tracks['players'], desc="Assigning ball to players")):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    #Draw output 
    ##Draw object Tracks
    
    print("🖌️ Drawing annotations on video frames...")
    
    output_video_frames = tracker.draw_annotations(video_frames,tracks)
    
    #save video
    print("💾 Saving output video...")
    
    save_video(output_video_frames, "output_videos/output_video.avi")
    
    print("✅ Terminé!")

if __name__=="__main__":
    main()
    