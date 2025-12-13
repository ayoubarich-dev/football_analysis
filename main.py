from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
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
    
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    print("📹 Estimating camera movement and adjusting object positions...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    print("🔄 Adjusting object positions for camera movement...")
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # View Trasnformer
    print("🔄 Transforming player and ball positions to top-down view...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate Ball Positions
    print("🧮 Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Speed and distance estimator
    print("🚀 Estimating speed and distance covered by players...")
    
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
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
    print("🏆 Assigning ball possession to players...")
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
    
    output_video_frames = tracker.draw_annotations(video_frames,tracks,team_ball_control)
    
    ## Draw Camera movement
    print("🎥 Drawing camera movement annotations on video frames...")
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
    ## Draw Speed and Distance
    print("🏃‍♂️ Drawing speed and distance annotations on video frames...")
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    #save video
    print("💾 Saving output video...")
    
    save_video(output_video_frames, "output_videos/output_video.avi")
    
    print("✅ Terminé!")

if __name__=="__main__":
    main()
    