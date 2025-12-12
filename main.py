from utils import read_video, save_video
from trackers import Tracker


def main():
    #read video
    print("🎬 Reading video...")
    video_frames = read_video("input_videos/08fd33_4.mp4")
    
    #initialize tracker
    print("🤖 Initializing tracker and getting object tracks...")
    
    tracker = Tracker("models/best.pt")
    
    print("⚡ Using stub for tracks...")
    
    tracks = tracker.get_object_trackes(video_frames,read_from_stub=True,stub_path="stubs/track_stubs.pkl")
    
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
    