model = load('model.h5')
path = input("Please input your testing file path: ")
pred = model.predict(load_img(path).reshape(1,200,200,1))

if pred.argmax() == 0.0:
	print("The model predicts that this x-ray does not display pneumonia")
else:
	print("The model predicts that this x-ray does display pneumonia")
	
