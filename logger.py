import csv
from datetime import datetime

class Logger:
    def __init__(self, unknown_csv='unknown.csv', failed_ocr_csv='failed_ocr.csv'):
        self.unknown_csv = unknown_csv
        self.failed_ocr_csv = failed_ocr_csv
        for path in [self.unknown_csv, self.failed_ocr_csv]:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'image_path', 'info'])

    def log_unknown(self, img_path, info):
        self._log(self.unknown_csv, img_path, info)

    def log_failed_ocr(self, img_path, info):
        self._log(self.failed_ocr_csv, img_path, info)

    def _log(self, path, img_path, info):
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), img_path, info])