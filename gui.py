import tkinter as tk
from tkinter import scrolledtext
from tkinter import font as tkFont
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt
import HMPM.optimism as op
from HMPM.hmpm import HMPM_MODEL
import sys, os
sys.stderr = open(os.devnull, 'w')

def submit_comment():
    # 获取输入框中的文本，并添加到评论列表中
    user_text = text_input.get("1.0", tk.END).strip()
    user_star = star_input.get("1.0", tk.END).strip()
    if user_text and user_star:  # 确保输入不是空的
        comments.append((user_text, user_star))
        text_input.delete("1.0", tk.END)  
        star_input.delete("1.0", tk.END)  
        print("保存的评论：", user_text)
        print("保存的评分：", user_star)
    # 弹窗显示：已保存！
    tk.messagebox.showinfo("提示", "已保存！")


def analyze_comments():
    # 拆分数据
    test_texts = [tup[0] for tup in comments]
    star_lis = np.array([tup[1] for tup in comments])

    # 优化解
    optimizer = op.weight_optimizer(model, tokenizer, test_texts, star_lis, device)
    final_weight_matrix = optimizer.optimize()

    # HMPM
    hmpm_model = HMPM_MODEL(final_weight_matrix)
    preference = hmpm_model.optimize()

    # 可视化
    comment_length = len(comments)
    aspect_lis =['environment', 'location', 'dish', 'service', 'price']
    colors = [op.generate_random_color() for _ in aspect_lis]
    # 图1，展示每个评论的情感分析结果
    if comment_length >= 4:
        fig1, axs1 = plt.subplots(2, 2, figsize=(6, 6))
        for i in range(2):
            for j in range(2):
                axs1[i, j].pie(final_weight_matrix[i * 2 + j] * 100, autopct='%1.1f%%', colors=colors, shadow=True, startangle=90)
                axs1[i, j].set_title(f'comment{i * 2 + j}, score:{star_lis[i * 2 + j]}')
    else:
        fig1, axs1 = plt.subplots(1, comment_length, figsize=(6, 6))
        for i in range(comment_length):
            axs1[i].pie(final_weight_matrix[i] * 100, autopct='%1.1f%%', colors=colors, shadow=True, startangle=90)
            axs1[i].set_title(f'comment{i}, score:{star_lis[i]}')
    fig1.legend(aspect_lis, loc='lower right')
    plt.subplots_adjust(right=0.8)
    # 图2，展示偏好分析结果
    fig2 = plt.Figure(figsize=(6, 6), dpi=100)
    ax2 = fig2.add_subplot(111)
    ax2.pie(preference * 100, autopct='%1.1f%%', colors=colors, shadow=True, startangle=90)
    ax2.set_title('preference')
    fig2.legend(aspect_lis, loc='lower right')
    plt.subplots_adjust(right=0.8)

    # 将饼状图显示在Tkinter窗口中
    canvas1 = FigureCanvasTkAgg(fig1, master=window)
    canvas_widget1 = canvas1.get_tk_widget()
    canvas_widget1.grid(row=2, column=0)
    canvas2 = FigureCanvasTkAgg(fig2, master=window)
    canvas_widget2 = canvas2.get_tk_widget()
    canvas_widget2.grid(row=2, column=1)

# 提示
def on_text_click(event):
    if text_input.get("1.0", "end-1c") == '请输入关于餐饮的评论...':
        text_input.delete("1.0", "end")
        text_input.config(fg='black')

def on_focusout(event):
    if text_input.get("1.0", "end-1c") == '':
        text_input.insert("1.0", '请输入关于餐饮的评论...')
        text_input.config(fg='grey')

def on_star_click(event):
    if star_input.get("1.0", "end-1c") == '请输入评论的综合评分,总分为3,可以精确一些比如2.25之类的...':
        star_input.delete("1.0", "end")
        star_input.config(fg='black')

def on_starfocusout(event):
    if star_input.get("1.0", "end-1c") == '':
        star_input.insert("1.0", '请输入评论的综合评分,总分为3,可以精确一些比如2.25之类的...')
        star_input.config(fg='grey')


# 准备参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_path = './newmodel/trained_model/model_bert_wwm.pth'
tokenizer_save_path = './newmodel/tokenizer1e-wwm'
tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)
model = BertForSequenceClassification.from_pretrained('./newmodel/chinese_wwm_ext_pytorch', num_labels=3)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)

# 创建主窗口
window = tk.Tk()
window.title("用户偏好分析")
window.configure(bg='#f0f0f0')  # 设置背景色

# 自定义字体
customFont = tkFont.Font(family="Helvetica", size=12)

# 存储评论的列表
comments = []

# 创建文本输入框
text_input = scrolledtext.ScrolledText(window, height=4, font=customFont)
text_input.grid(row=0, column=0, sticky="ew")
star_input = scrolledtext.ScrolledText(window, height=2, font=customFont)
star_input.grid(row=1, column=0, sticky="ew")
text_input.insert("1.0", '请输入关于餐饮的评论...')
text_input.config(fg='grey')
text_input.bind('<FocusIn>', on_text_click)
text_input.bind('<FocusOut>', on_focusout)
star_input.insert("1.0", '请输入评论的综合评分,总分为3,可以精确一些比如2.25之类的...')
star_input.config(fg='grey')
star_input.bind('<FocusIn>', on_star_click)
star_input.bind('<FocusOut>', on_starfocusout)

# 创建提交按钮
submit_button = tk.Button(window, text="提交评论", command=submit_comment, bg="#4caf50")
submit_button.grid(row=0, column=1, sticky="ew")

# 创建分析按钮
analyze_button = tk.Button(window, text="分析", command=analyze_comments, bg="#2196f3")
analyze_button.grid(row=1, column=1, sticky="ew")

# 布局
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=1)
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)

window.mainloop()
