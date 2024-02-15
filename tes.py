import string

def loai_bo_dau_cau(input_string):
    # Sử dụng string.punctuation để lấy tất cả các dấu câu
    punctuation_set = set(string.punctuation)

    # Lọc ra các ký tự không phải dấu câu từ chuỗi đầu vào
    result = ''.join(char for char in input_string if char not in punctuation_set)

    return result

# # Ví dụ
# chuoi_dau_vao = "Đây là. một chuỗi, có dấu câu! Liệu có thể loại bỏ chúng không?"
# chuoi_ket_qua = loai_bo_dau_cau(chuoi_dau_vao)

# print("Chuỗi đầu vào:", chuoi_dau_vao)
# print("Chuỗi kết quả:", chuoi_ket_qua)
