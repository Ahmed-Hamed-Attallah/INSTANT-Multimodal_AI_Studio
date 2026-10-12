class LayoutAnalyzer:
    @staticmethod
    def sort_by_reading_order(detections: list[dict], y_tolerance: int = 10) -> list[dict]:
        """
        Sorts bounding boxes top-to-bottom, left-to-right.
        """
        def box_key(item):
            # item['box'][0] is top-left coordinate [x, y]
            x, y = item["box"][0]
            # Quantize y to cluster roughly identical lines
            return (int(y // y_tolerance), x)

        return sorted(detections, key=box_key)