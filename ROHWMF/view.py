# ！usr/bin/python
# -*- coding:utf-8 -*-#
# @date:2019/11/6 10:25
# @name:view
# @author:TDYe
from PIL import Image
from .utils.screen import *
from .utils.cnnModel import *
from .utils.calculator import *
from django.shortcuts import render
from .utils.imageProcessing import *
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

meta = r"D:\MyProjects\ROHWMF\ROHWMF\static\model\model-120.meta"
path = r"D:\MyProjects\ROHWMF\ROHWMF\static\model"


def index(request):
	return render(request, 'index.html')


def save_img(img_arr: np.ndarray, file_path: str) -> None:
	img = Image.fromarray(img_arr, 'L')
	img.save(file_path)


@csrf_exempt
def getResult(request):
	img_str = request.POST["img_data"]
	img_arr = np.array(img_str.split(',')).reshape(200, 1000, 4).astype(np.uint8)
	binary_img_arr = img_arr[:, :, 3]
	save_img(binary_img_arr, "./target.png")
	data = cv2.imread('./target.png', 2)
	data = 255 - data
	images = get_image_cuts(data, is_data=True, n_lines=1, data_needed=True)
	equation = ''
	cnnModel = Model()
	cnnModel.load_model(meta, path)
	digits = list(cnnModel.predict(images))
	for d in digits:
		equation += SYMBOL[d]
	print("数据驱动分析结果：" + equation)
	equation = screen(equation)
	print("知识驱动分析结果：" + equation)
	result = calculate(equation)
	return JsonResponse({"status": "{} = {}".format(equation, result)}, safe=False)