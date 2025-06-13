import cv2
import numpy as np
import sys
sys.path.append('../')
# import constants
# from utils import (
#     convert_meters_to_pixel_distance,
#     convert_pixel_distance_to_meters,
#     get_foot_position,
#     get_closest_keypoint_index,
#     get_height_of_bbox,
#     measure_xy_distance,
#     get_center_of_bbox,
#     measure_distance
# )
from tennis_analysis.utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)
import constants


class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(
            meters,
            constants.DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0] * 28

        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1]
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3]
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7]
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
        drawing_key_points[25] = drawing_key_points[17]
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        net_start = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start, net_end, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_mini_court(self, frames):
        return [self.draw_court(self.draw_background_rectangle(frame)) for frame in frames]

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self, object_position, closest_key_point, closest_key_point_index,
                                   player_height_in_pixels, player_height_in_meters):
        dx_pix, dy_pix = measure_xy_distance(object_position, closest_key_point)
        dx_m = convert_pixel_distance_to_meters(dx_pix, player_height_in_meters, player_height_in_pixels)
        dy_m = convert_pixel_distance_to_meters(dy_pix, player_height_in_meters, player_height_in_pixels)
        dx_pix_mini = self.convert_meters_to_pixels(dx_m)
        dy_pix_mini = self.convert_meters_to_pixels(dy_m)
        key_x = self.drawing_key_points[closest_key_point_index * 2]
        key_y = self.drawing_key_points[closest_key_point_index * 2 + 1]
        return (key_x + dx_pix_mini, key_y + dy_pix_mini)

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        player_heights = {
          1: constants.PLAYER_1_HEIGHT_METERS,
          2: constants.PLAYER_2_HEIGHT_METERS
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                height = player_heights.get(player_id)
                if height is None:
                    continue

                foot_position = get_foot_position(bbox)
                key_index = get_closest_keypoint_index(foot_position, original_court_key_points, [0, 2, 12, 13])
                key_point = (original_court_key_points[key_index * 2], original_court_key_points[key_index * 2 + 1])

                i_min = max(0, frame_num - 20)
                i_max = min(len(player_boxes), frame_num + 50)
                heights = [get_height_of_bbox(player_boxes[i][player_id]) for i in range(i_min, i_max)]
                max_height_pixels = max(heights)

                pos = self.get_mini_court_coordinates(foot_position, key_point, key_index, max_height_pixels, height)
                output_player_bboxes_dict[player_id] = pos

            output_player_boxes.append(output_player_bboxes_dict)

        # Convert ball position for this frame
            if frame_num < len(ball_boxes):
                ball_dict = ball_boxes[frame_num]
                if ball_dict and 1 in ball_dict:
                    ball_bbox = ball_dict[1]
                    ball_position = get_center_of_bbox(ball_bbox)
                    ball_key_index = get_closest_keypoint_index(ball_position, original_court_key_points, [0, 2, 12, 13])
                    ball_key_point = (original_court_key_points[ball_key_index * 2], original_court_key_points[ball_key_index * 2 + 1])
                    ball_pos = self.get_mini_court_coordinates(ball_position, ball_key_point, ball_key_index, 40, 0.25)


                    output_ball_boxes.append({1: ball_pos})
                else:
                    output_ball_boxes.append({})
            else:
                output_ball_boxes.append({})

        return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        for frame_num in range(len(frames)):
            if frame_num >= len(positions):
                continue
            frame = frames[frame_num]
            for _, position in positions[frame_num].items():
                x, y = map(int, position)
                x = min(max(x, self.start_x), self.end_x - 1)
                y = min(max(y, self.start_y), self.end_y - 1)

                cv2.circle(frame, (x, y), 5, color, -1)

        print(f"Mini-court dot (color {color}): Frame {frame_num}, position: {position}")
        cv2.putText(frame, "•", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)


    
        return frames  # ✅ return after the loop finishes


