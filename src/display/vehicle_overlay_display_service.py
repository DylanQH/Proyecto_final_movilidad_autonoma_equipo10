from controller import Display
import cv2


class VehicleOverlayDisplayService:
    """
    Draws a camera HUD on the vehicle Webots Display.
    """

    def __init__(self, robot, display_name, fallback_names=("display",)):
        self.display = robot.getDevice(display_name)
        self.display_name = display_name

        if self.display is None:
            for fallback_name in fallback_names:
                self.display = robot.getDevice(fallback_name)
                if self.display is not None:
                    self.display_name = fallback_name
                    break

        if self.display is None:
            print(f"[VehicleOverlay][WARNING] Vehicle display not found: {display_name}")
            self.width = 0
            self.height = 0
            return

        try:
            self.width = self.display.getWidth()
            self.height = self.display.getHeight()
        except Exception:
            self.width = 0
            self.height = 0

        print(
            f"[VehicleOverlay] Vehicle camera overlay display '{self.display_name}' "
            f"enabled ({self.width}x{self.height})."
        )

    def update(
        self,
        camera_image,
        state,
        lidar_distance=None,
        pedestrian_detected=False,
        pedestrian_alert=False,
        brake_active=False,
    ):
        if self.display is None or camera_image is None:
            return

        img = camera_image[:, :, :3].copy()

        if self.width > 0 and self.height > 0:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

        h, w = img.shape[:2]

        self._draw_status_bar(
            img=img,
            state=state,
            lidar_distance=lidar_distance,
        )
        self._draw_led(
            img=img,
            center=(w - 70, 18),
            label="PED",
            color_on=(0, 255, 255),
            color_off=(0, 75, 75),
            active=pedestrian_detected,
        )
        self._draw_led(
            img=img,
            center=(w - 70, 34),
            label="ALR",
            color_on=(0, 220, 255),
            color_off=(0, 55, 65),
            active=pedestrian_alert,
        )
        self._draw_led(
            img=img,
            center=(w - 70, 50),
            label="BRK",
            color_on=(0, 0, 255),
            color_off=(0, 0, 75),
            active=brake_active,
        )

        img_rgb = img[:, :, ::-1].copy()
        image_ref = self.display.imageNew(
            img_rgb.tobytes(),
            Display.RGB,
            width=img_rgb.shape[1],
            height=img_rgb.shape[0],
        )
        self.display.imagePaste(image_ref, 0, 0, False)
        self.display.imageDelete(image_ref)

    def _draw_status_bar(self, img, state, lidar_distance):
        h, _ = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 18), (img.shape[1], h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        if lidar_distance is None:
            lidar_text = "L:--"
        else:
            lidar_text = f"L:{lidar_distance:.1f}m"

        text = f"{self._short_state(state)}  {lidar_text}"
        cv2.putText(
            img,
            text,
            (4, h - 5),
            cv2.FONT_HERSHEY_PLAIN,
            0.75,
            (255, 255, 255),
            1,
        )

    def _draw_led(self, img, center, label, color_on, color_off, active):
        x, y = center
        color = color_on if active else color_off
        cv2.circle(img, (x, y), 5, color, -1)
        cv2.putText(
            img,
            label,
            (x + 8, y + 4),
            cv2.FONT_HERSHEY_PLAIN,
            0.65,
            color,
            1,
        )

    def _short_state(self, state):
        if state == "BEHAVIORAL_CLONING_NORMAL":
            return "BC"
        if state == "MANUAL_EMERGENCY_BRAKE":
            return "BRAKE"
        if state == "KEYBOARD_MANUAL_DRIVE":
            return "KEY"
        if state.startswith("INTERSECTION"):
            return "XING"
        if state.startswith("AVOID"):
            return "AVOID"
        if state.startswith("RADAR"):
            return "RADAR"
        return state[:8]
