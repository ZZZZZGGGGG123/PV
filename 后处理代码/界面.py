from PySide2.QtWidgets import QApplication, QMessageBox, QMainWindow
from PySide2.QtUiTools import QUiLoader

class Stats:

    def __init__(self):
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button_image_extraction, self.ui.button_image_binarization 等等
        self.ui = QUiLoader().load(r'E:\code\unet-pytorch-main\后处理代码\main.ui.xml')

        self.ui.button_image_extraction.clicked.connect(self.handle_image_extraction)
        self.ui.button_image_binarization.clicked.connect(self.handle_image_binarization)
        self.ui.button_image_crop.clicked.connect(self.handle_image_crop)
        self.ui.button_image_stitching.clicked.connect(self.handle_image_stitching)
        self.ui.button_filter_target_image.clicked.connect(self.handle_filter_target_image)
        self.ui.button_raster_to_vector.clicked.connect(self.handle_raster_to_vector)
        self.ui.button_batch_raster_to_vector.clicked.connect(self.handle_batch_raster_to_vector)
        self.ui.button_image_compression.clicked.connect(self.handle_image_compression)

    def handle_image_extraction(self):
        pass

    def handle_image_binarization(self):
        pass

    def handle_image_crop(self):
        pass

    def handle_image_stitching(self):
        pass

    def handle_filter_target_image(self):
        pass

    def handle_raster_to_vector(self):
        pass

    def handle_batch_raster_to_vector(self):
        pass

    def handle_image_compression(self):
        pass

app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec_()
