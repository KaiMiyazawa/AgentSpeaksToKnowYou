import pandas as pd
import numpy as  np
from sklearn.metrics.pairwise import cosine_similarity
import re


# アンケートデータの整形============================================================================================================
    # アンケートのデータ読み込み
df = pd.read_csv("性格特性と文章の捉え方についての調査_2024年7月22日_03.26.csv", header=0)

df_result = df.drop(df.index[0:2])
df_result = df_result.drop(df.columns[0:19], axis=1)

df_int = df_result.astype(int)
df_int["Openness"] = df_int["Q3.1_5"] + (8 - df_int["Q3.1_10"])
df_int["Conscientiousness"] = df_int["Q3.1_3"] + (8 - df_int["Q3.1_8"])
df_int["Extraversion"] = df_int["Q3.1_1"] + (8 - df_int["Q3.1_6"])
df_int["Agreeableness"] = (8 - df_int["Q3.1_2"]) + df_int["Q3.1_7"]
df_int["Neuroticism"] = df_int["Q3.1_4"] + df_int["Q3.1_9"]

test_factor5 = [0.1928376, 0.13944813, 0.4152467, 0.26870754, 0.40584159]
#Factor1: [0.1928376]
#Factor2: [0.13944813]
#Factor3: [0.4152467]
#Factor4: [0.26870754]
#Factor: [0.40584159]

test_factor5 = [0.29718495, 0.21891212, 0.33789448, 0.21065025, 0.25343672]
#Factor1: [0.29718495]
#Factor2: [0.21891212]
#Factor: [0.33789448]
#Factor4: [0.21065025]
#Factor5: [0.25343672]
# ================================================================================================================================





# 文章のデータ読み込み・整形==============================================================================
df_sentences = pd.read_csv("性格診断_出力(小説) - シート1.csv", header=None)
columns = ["文章", "作者"]
df_sentences.columns = columns

df_sentences.head()
# =====================================================================================================





# テキストの方のデータを比較===============================================================================================

test_bigfive = [0.4, 0.9, 0.9, 0.6, 0.1]


text_columns = [col for col in df_int.columns if col.endswith("_6")]
text_columns.remove("Q3.1_6")
text_columns.extend(["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"])

df_symp = df_int[text_columns]

# 最後の5列の値を取り出して辞書を作成
last_5_columns = df_symp.columns[-5:]
result_dict = {index: row[last_5_columns].tolist() for index, row in df_symp.iterrows()}

 # コサイン類似度を計算
bigfive_columns = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
similarities = cosine_similarity([test_bigfive], df_symp[bigfive_columns])[0]

# 類似度が最も高い3行を取得
top_3_indices = similarities.argsort()[-3:][::-1]
top_3_rows = df_symp.iloc[top_3_indices]

# '_6'で終わる列の平均を計算
columns_ending_with_6 = [col for col in df_symp.columns if col.endswith('_6')]
mean_values = top_3_rows[columns_ending_with_6].mean()

# 最も平均値が高い列名を取得
max_mean_column = mean_values.idxmax()

# 正規表現を使用してすべての数値部分を抽出
matches = re.findall(r'\d+', max_mean_column)
if matches:
    # リストから数値部分を選択
    most_similar_question_number = int(matches[-2])
else:
    print("No number found in the string")

sentence_text = df_sentences.iloc[most_similar_question_number - 1, 0]
author_text = df_sentences.iloc[most_similar_question_number - 1, 1]

print(f"sentence: {sentence_text}, author: {author_text}")

# ========================================================================================================





# 音声の方を比較する==================================================================================




# 必要な列名をフィルタリング
required_columns = [col for col in df_int.columns if col.startswith('Q5.')]

# 新しいデータフレームを作成
df_factors = df_int[required_columns]

unrequired_columns = [col for col in df_factors if col.endswith("_6")]

df_factors = df_factors.drop(unrequired_columns, axis=1)

# 文章ごと、factor5ごとに指標の平均値を算出する
factors_means = df_factors.mean()

# 辞書を作成(問番号がキー、平均値のリストがバリュー)
means_dict = {}

for index, value in factors_means.items():
    key = index.rsplit('_', 1)[0]
    if key not in means_dict:
        means_dict[key] = []
    means_dict[key].append(value)

## 結果を確認
#for key, value in means_dict.items():
#    print(f"{key}: {value}")
#    tmp = re.findall(r'\d+', key)
#    if tmp:
#        # リストから最後の数値部分を選択
#        ttt = int(tmp[-1])
#    else:
#        print("No number found in the string")

#    # 文章と作者の抽出
#    s = df_sentences.iloc[ttt - 1, 0]
#    a = df_sentences.iloc[ttt - 1, 1]
#    print(f"sentence: {s}, author: {a}")


# コサイン類似度を計算し、最も類似している番号を見つける
max_similarity = -1
most_similar_key = None

# reshape to 2D array for cosine_similarity function
test_vector = np.array(test_factor5).reshape(1, -1)

for key, values in means_dict.items():
    values_vector = np.array(values).reshape(1, -1)
    similarity = cosine_similarity(test_vector, values_vector)[0][0]
    if similarity > max_similarity:
        max_similarity = similarity
        most_similar_key = key

# 正規表現を使用してすべての数値部分を抽出
matches = re.findall(r'\d+', most_similar_key)
if matches:
    # リストから最後の数値部分を選択
    most_similar_question_number = int(matches[-1])
else:
    print("No number found in the string")

# 文章と作者の抽出
sentence_voice = df_sentences.iloc[most_similar_question_number - 1, 0]
author_voice = df_sentences.iloc[most_similar_question_number - 1, 1]

print(f"sentence: {sentence_voice}, author: {author_voice}")

# ====================================================================================================