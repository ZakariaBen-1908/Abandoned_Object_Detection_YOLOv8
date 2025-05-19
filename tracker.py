import math
import time


class ObjectTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will fincrease by one
        self.id_count = 0

        # Dictionary to store the time when an object is first detected as stationary
        self.abandoned_time = {}

        # Dictionary to store the class ID of each tracked object
        self.object_classes = {}

        # Threshold for considering an object as abandoned (in seconds)
        self.abandoned_threshold = 5  # 5 seconds

    def update(self, objects_rect, class_ids=None):
        # Objects boxes and ids
        objects_bbs_ids = []
        abandoned_objects = []

        # Get the current time
        current_time = time.time()

        # Get center point of new object
        for idx, rect in enumerate(objects_rect):
            x, y, w, h = rect
            cx = (x + x + w) / 2
            cy = (y + y + h) / 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                distance = math.hypot(cx - pt[0], cy - pt[1])

                # If the object is close to its previous position (stationary)
                if distance < 25:
                    # Update the center point
                    self.center_points[id] = (cx, cy)

                    # Check if the object is stationary
                    if id in self.abandoned_time:
                        # If the object has moved, reset its timer
                        if distance > 1:
                            self.abandoned_time[id] = current_time
                        # If the object is stationary for more than the threshold, mark it as abandoned
                        elif current_time - self.abandoned_time[id] >= self.abandoned_threshold:
                            # Include the class ID in the abandoned object data
                            class_id = self.object_classes.get(id, None)
                            abandoned_objects.append([id, x, y, w, h, distance, class_id])
                    else:
                        # Initialize the timer for the object
                        self.abandoned_time[id] = current_time

                    objects_bbs_ids.append([x, y, w, h, id, distance])
                    same_object_detected = True
                    break

            # If a new object is detected, assign an ID to it
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                self.abandoned_time[self.id_count] = current_time  # Initialize the timer
                if class_ids is not None:
                    self.object_classes[self.id_count] = class_ids[idx]  # Store the class ID
                objects_bbs_ids.append([x, y, w, h, self.id_count, None])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        new_abandoned_time = {}
        new_object_classes = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, _ = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

            if object_id in self.abandoned_time:
                new_abandoned_time[object_id] = self.abandoned_time[object_id]

            if object_id in self.object_classes:
                new_object_classes[object_id] = self.object_classes[object_id]

        # Update dictionaries with IDs not used removed
        self.center_points = new_center_points.copy()
        self.abandoned_time = new_abandoned_time.copy()
        self.object_classes = new_object_classes.copy()

        return objects_bbs_ids, abandoned_objects