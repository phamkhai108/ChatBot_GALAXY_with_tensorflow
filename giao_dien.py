import tkinter as tk
from tkinter import Text, END

# Import hàm từ file khác
from chat import chat

class SimpleChatApp:
    def __init__(self, master):
        self.master = master
        master.title("chatbot GALAXY")

        # Tạo Text widget để hiển thị cuộc trò chuyện và thiết lập state là DISABLED
        self.chat_display = Text(master, width=70, height=20, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_display.grid(row=0, column=0, sticky="nsew")

        # Tạo Entry widget để người dùng nhập vào
        self.input_entry = tk.Entry(master, width=40, font=('Arial', 14)) 
        self.input_entry.grid(row=1, column=0, pady=10, padx=10, sticky="ew")
        # Tạo nút "Send" để gửi tin nhắn
        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, pady=10, padx=10, sticky="ew")

        # Tạo tag để đặt màu cho dòng nhập của người dùng và câu trả lời của bot
        self.chat_display.tag_configure("user_tag", foreground="blue")
        self.chat_display.tag_configure("bot_tag", foreground="red")

        # Đặt trọng số cột và hàng để widget có thể mở rộng
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        master.rowconfigure(1, weight=0)  # Không mở rộng hàng này

    def send_message(self):
        # Lấy giá trị từ ô nhập liệu
        user_input = self.input_entry.get()

        # Hiển thị dòng nhập của người dùng
        self.display_user_message(user_input)
        # Gọi hàm xử lý từ file khác và nhận kết quả
        result = chat(user_input)

        # Hiển thị phản hồi của chatbot
        self.display_chatbot_message(result)

        # Xóa nội dung trong ô nhập liệu
        self.input_entry.delete(0, tk.END)

    def display_user_message(self, message):
        # Thiết lập state của Text widget thành NORMAL để có thể thêm dòng nhập của người dùng
        self.chat_display.config(state=tk.NORMAL)
        
        # Thêm dòng nhập của người dùng vào Text widget ở vị trí con trỏ hiện tại với tag "user_tag"
        self.chat_display.insert(tk.INSERT, "You: " + message + "\n", "user_tag")
        
        # Thiết lập state của Text widget thành DISABLED để không thể nhập liệu vào
        self.chat_display.config(state=tk.DISABLED)
        
        # Cuộn xuống để hiển thị tin nhắn mới nhất
        self.chat_display.see(tk.END)

    def display_chatbot_message(self, message):
        # Thiết lập state của Text widget thành NORMAL để có thể thêm phản hồi của chatbot
        self.chat_display.config(state=tk.NORMAL)
        
        # Thêm phản hồi của chatbot vào Text widget ở vị trí con trỏ hiện tại với tag "bot_tag"
        self.chat_display.insert(tk.INSERT, "GALAXY: " + message + "\n\n", "bot_tag")
        
        # Thiết lập state của Text widget thành DISABLED để không thể nhập liệu vào
        self.chat_display.config(state=tk.DISABLED)
        
        # Cuộn xuống để hiển thị tin nhắn mới nhất
        self.chat_display.see(tk.END)

# Khởi tạo và hiển thị giao diện
root = tk.Tk()
app = SimpleChatApp(root)

# Đặt kích thước mặc định cho cửa sổhe
root.geometry("1000x600")
root.mainloop()
