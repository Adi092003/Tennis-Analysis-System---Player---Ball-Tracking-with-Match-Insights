from utils import (
    read_video,
    save_video,
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters
)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy


def main():
    # Read Video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')

    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player_detections.pkl"
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/ball_detections.pkl"
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Line Detection
    court_model_path = "models\keypoints_model .pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Filter players dynamically
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Mini Court setup
    mini_court = MiniCourt(video_frames[0])

    # Detect shot frames
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert to mini court coordinates
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints
    )

    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,
        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps

        player_positions = player_mini_court_detections[start_frame] if start_frame < len(player_mini_court_detections) else {}
        ball_position = ball_mini_court_detections[start_frame].get(1) if start_frame < len(ball_mini_court_detections) else None

        player1_pos = player_positions.get(1)
        player2_pos = player_positions.get(2)

        if ball_position is None or player1_pos is None or player2_pos is None:
            continue

        player_shot_ball = min(
            [1, 2],
            key=lambda player_id: measure_distance(player1_pos if player_id == 1 else player2_pos, ball_position)
        )
        opponent_player_id = 1 if player_shot_ball == 2 else 2

        # Distance and speed calculations
        ball_start = ball_mini_court_detections[start_frame].get(1)
        ball_end = ball_mini_court_detections[end_frame].get(1)
        if ball_start is None or ball_end is None:
            continue

        distance_ball_pixels = measure_distance(ball_start, ball_end)
        distance_ball_meters = convert_pixel_distance_to_meters(
            distance_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        speed_ball = distance_ball_meters / ball_shot_time_in_seconds * 3.6

        opponent_start = player_mini_court_detections[start_frame].get(opponent_player_id)
        opponent_end = player_mini_court_detections[end_frame].get(opponent_player_id)
        if opponent_start is None or opponent_end is None:
            continue

        distance_opponent_pixels = measure_distance(opponent_start, opponent_end)
        distance_opponent_meters = convert_pixel_distance_to_meters(
            distance_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        speed_opponent = distance_opponent_meters / ball_shot_time_in_seconds * 3.6

        current_stats = deepcopy(player_stats_data[-1])
        current_stats['frame_num'] = start_frame
        current_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_ball
        current_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_ball
        current_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_opponent
        current_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_opponent

        player_stats_data.append(current_stats)

    player_stats_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_df = pd.merge(frames_df, player_stats_df, on='frame_num', how='left').ffill()

    for i in [1, 2]:
        player_stats_df[f'player_{i}_average_shot_speed'] = player_stats_df[f'player_{i}_total_shot_speed'] / player_stats_df[f'player_{i}_number_of_shots'].replace(0, 1)
        player_stats_df[f'player_{i}_average_player_speed'] = player_stats_df[f'player_{i}_total_player_speed'] / player_stats_df[f'player_{i}_number_of_shots'].replace(0, 1)

    # Draw everything
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
   # First draw the background and court lines
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

# THEN draw ball and player dots
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)

    output_video_frames = draw_player_stats(output_video_frames, player_stats_df)

    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if len(output_video_frames) > 0:
        save_video(output_video_frames, "output_videos/output_video.avi")
    else:
        print("No frames generated to save.")
    
    print(f"Frame {start_frame}: Player1 = {player1_pos}, Player2 = {player2_pos}, Ball = {ball_position}")



if __name__ == "__main__":
    main()
