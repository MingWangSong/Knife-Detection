import os

if __name__ == "__main__":
    for file in list(os.listdir("D:\文件\桂电\毕业论文\宋\圆盘刀检测小论文\投稿\pdf")):
        if file.endswith("pdf"):
            os.system('pdfcrop %s %s' % (file, file))
            os.system('pdftops %s %s' % (file, file[:-4] + ".eps"))
