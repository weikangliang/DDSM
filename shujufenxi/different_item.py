# coding=utf-8
file_test1_path = '../dataset/Sports/local_train_sample_sorted_by_time'
file_test2_path = '../dataset/Sports/local_test_sample_sorted_by_time'

# 读取文件并提取物品ID
def read_user_ids(file_path):
    user_ids = set()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) > 2:
                user_ids.add(parts[2])
    return user_ids

# 读取test1和test2中的物品ID
user_ids_test1 = read_user_ids(file_test1_path)
user_ids_test2 = read_user_ids(file_test2_path)
print(u"训练集中的物品数：",len(user_ids_test1))
print(u"测试集中的物品数：",len(user_ids_test2))
difference = user_ids_test2 - user_ids_test1
def read_different_user_ids(file_path,difference):
    count_diff,count_all = 0,0
    with open(file_path, 'r') as file:
        for line in file:
            count_all+=1
            parts = line.strip().split('\t')
            if len(parts) > 2 and parts[2] in difference:
                count_diff+=1
    return count_all,count_diff


# 计算在test2中但不在test1中的物品数量
unique_users_in_test2_not_in_test1 = len(difference)
print(u"差异的物品数：",unique_users_in_test2_not_in_test1)


count_all,count_diff = read_different_user_ids(file_test2_path,difference)
print(u"测试集总的样本数：",count_all)
print(u"差异的物品所占的样本行数：",count_diff)