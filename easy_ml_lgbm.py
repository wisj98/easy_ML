import pandas as pd
import numpy as np

from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import xgboost
import sklearn

import optuna

raw_train = pd.read_csv("test_data/train.csv")
raw_test = pd.read_csv("test_data/test.csv")

# FE
def FE_1(train_, test_):
    train = train_.copy()
    test = test_.copy()

    train["시술 당시 나이__"] = train["시술 당시 나이"].apply(lambda x: (int(x[1:3]) + int(x[4:6]))/2 if x[2] != "수" else -1)
    test["시술 당시 나이__"] = test["시술 당시 나이"].apply(lambda x: (int(x[1:3]) + int(x[4:6]))/2 if x[2] != "수" else -1)

    train = train.drop(columns = ["시술 당시 나이"])
    test = test.drop(columns = ["시술 당시 나이"])

    return train, test

def FE_2(train_, test_):
    train = train_.copy()
    test = test_.copy()

    train["배아 생성 주요 이유"] = train["배아 생성 주요 이유"].apply(lambda x: [i.replace(" ", "") for i in x.split(",")] if type(x) != float else ["DI"])
    train["현재시술용"] = train["배아 생성 주요 이유"].apply(lambda x: "현재시술용" in x)
    train["배아저장용"] = train["배아 생성 주요 이유"].apply(lambda x: "배아저장용" in x)
    train["기증용"] = train["배아 생성 주요 이유"].apply(lambda x: "기증용" in x)
    train["DI"] = train["배아 생성 주요 이유"].apply(lambda x: "di" in x)

    test["배아 생성 주요 이유"] = test["배아 생성 주요 이유"].apply(lambda x: [i.replace(" ", "") for i in x.split(",")] if type(x) != float else ["DI"])
    test["현재시술용"] = test["배아 생성 주요 이유"].apply(lambda x: "현재시술용" in x)
    test["배아저장용"] = test["배아 생성 주요 이유"].apply(lambda x: "배아저장용" in x)
    test["기증용"] = test["배아 생성 주요 이유"].apply(lambda x: "기증용" in x)
    test["DI"] = test["배아 생성 주요 이유"].apply(lambda x: "di" in x)

    train = train.drop(columns = "배아 생성 주요 이유")
    test = test.drop(columns = "배아 생성 주요 이유")
    return train, test

def FE_3(train_, test_):
    train = train_.copy()
    test = test_.copy()

    train["임신 시도 또는 마지막 임신 경과 연수"] = train["임신 시도 또는 마지막 임신 경과 연수"].fillna(-1)
    test["임신 시도 또는 마지막 임신 경과 연수"] = test["임신 시도 또는 마지막 임신 경과 연수"].fillna(-1)

    return train, test

def FE_4(train_, test_):
    train = train_.copy()
    test = test_.copy()

    train["특정 시술 유형"] = train["특정 시술 유형"].apply(lambda x: x.replace(" ", "").replace(":", "/") if type(x) != float else "")
    train["특정 시술 유형"] = train["특정 시술 유형"].apply(lambda x: x.split("/"))
    train["ICSI"] = train["특정 시술 유형"].apply(lambda x: "ICSI" in x)
    train["AH"] = train["특정 시술 유형"].apply(lambda x: "AH" in x)
    train["BLASTOCYST"] = train["특정 시술 유형"].apply(lambda x: "BLASTOCYST" in x)
    train["IVF"] = train["특정 시술 유형"].apply(lambda x: "IVF" in x)
    train["Unknown"] = train["특정 시술 유형"].apply(lambda x: "Unknown" in x)

    train["IUI"] = train["특정 시술 유형"].apply(lambda x: "IUI" in x)
    train["ICI"] = train["특정 시술 유형"].apply(lambda x: "ICI" in x)
    train["GIFT"] = train["특정 시술 유형"].apply(lambda x: "GIFT" in x)
    train["Generic DI"] = train["특정 시술 유형"].apply(lambda x: "Generic DI" in x)
    train["IVI"] = train["특정 시술 유형"].apply(lambda x: "IVI" in x)

    test["특정 시술 유형"] = test["특정 시술 유형"].apply(lambda x: x.replace(" ", "").replace(":", "/") if type(x) != float else "")
    test["특정 시술 유형"] = test["특정 시술 유형"].apply(lambda x: x.split("/"))
    test["ICSI"] = test["특정 시술 유형"].apply(lambda x: "ICSI" in x)
    test["AH"] = test["특정 시술 유형"].apply(lambda x: "AH" in x)
    test["BLASTOCYST"] = test["특정 시술 유형"].apply(lambda x: "BLASTOCYST" in x)
    test["IVF"] = test["특정 시술 유형"].apply(lambda x: "IVF" in x)
    test["Unknown"] = test["특정 시술 유형"].apply(lambda x: "Unknown" in x)

    test["IUI"] = test["특정 시술 유형"].apply(lambda x: "IUI" in x)
    test["ICI"] = test["특정 시술 유형"].apply(lambda x: "ICI" in x)
    test["GIFT"] = test["특정 시술 유형"].apply(lambda x: "GIFT" in x)
    test["Generic DI"] = test["특정 시술 유형"].apply(lambda x: "Generic DI" in x)
    test["IVI"] = test["특정 시술 유형"].apply(lambda x: "IVI" in x)

    
    train = train.drop(columns = "특정 시술 유형")
    test = test.drop(columns = "특정 시술 유형")

    return train, test

def FE_5(train_, test_):
    train = train_.copy()
    test = test_.copy()

    for col in ["총 시술 횟수__", "클리닉 내 총 시술 횟수__", "IVF 시술 횟수__", "DI 시술 횟수__", "총 임신 횟수__", "IVF 임신 횟수__", "DI 임신 횟수__", "총 출산 횟수__", "IVF 출산 횟수__", "DI 출산 횟수__"]:
        train[col] = train[col].apply(lambda x: int(x[0]))
        test[col] = test[col].apply(lambda x: int(x[0]))

    train = train.drop(columns = ["총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수", "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"])
    test = test.drop(columns = ["총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수", "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"])
    
    return train, test

def FE_6(train_, test_):
    train = train_.copy()
    test = test_.copy()

    for col in ["배아 이식 경과일", "난자 혼합 경과일", "배아 해동 경과일"]:
        train[col] = train[col].fillna(-1)
        test[col] = test[col].fillna(-1)
    
    return train, test

def FE_7(train_, test_):
    train = train_.copy()
    test = test_.copy()

    train["시술 당시 나이_"] = train["시술 당시 나이"].apply(lambda x: (int(x[1:3]) + int(x[4:6]))/2 if x[2] != "수" else -1)
    test["시술 당시 나이_"] = test["시술 당시 나이"].apply(lambda x: (int(x[1:3]) + int(x[4:6]))/2 if x[2] != "수" else -1)

    train["난자 기증자 나이"] = train.apply(lambda row: int(row["난자 기증자 나이"][1:3]) if row["난자 기증자 나이"] != "알 수 없음" else row["시술 당시 나이_"], axis = 1)
    test["난자 기증자 나이"] = test.apply(lambda row: int(row["난자 기증자 나이"][1:3]) if row["난자 기증자 나이"] != "알 수 없음" else row["시술 당시 나이_"], axis = 1)

    train = train.drop(columns = "시술 당시 나이_")
    test = test.drop(columns = "시술 당시 나이_")

    return train, test

def FE_8(train_, test_):
    train = train_.copy()
    test = test_.copy()

    train["불임 원인 총 합"] = 0
    for col in [i for i in train.columns if i[:5] == "불임 원인"]:
        train["불임 원인 총 합"] = train["불임 원인 총 합"] + train[col]

    test["불임 원인 총 합"] = 0
    for col in [i for i in test.columns if i[:5] == "불임 원인"]:
        test["불임 원인 총 합"] = test["불임 원인 총 합"] + test[col]

    return train, test

def FE_9(train_, test_):
    train = train_.copy()
    test = test_.copy()

    for col in ["IVF 시술 횟수", "IVF 출산 횟수"]:
        train[col+"_"] = train[col].apply(lambda x: int(x[0]))
        test[col+"_"] = test[col].apply(lambda x: int(x[0]))

    train["기존 IVF 성공 확률"] = train.apply(lambda row: row["IVF 출산 횟수_"]/row["IVF 시술 횟수_"] if row["IVF 시술 횟수_"] != 0 else 0, axis = 1)
    test["기존 IVF 성공 확률"] = test.apply(lambda row: row["IVF 출산 횟수_"]/row["IVF 시술 횟수_"] if row["IVF 시술 횟수_"] != 0 else 0, axis = 1)

    train = train.drop(columns = ["IVF 출산 횟수_", "IVF 시술 횟수_"])
    test = test.drop(columns = ["IVF 출산 횟수_", "IVF 시술 횟수_"])

    return train, test

def FE_10(train_, test_):
    train = train_.copy()
    test = test_.copy()

    for col in ["IVF 시술 횟수", "IVF 출산 횟수"]:
        train[col+"_"] = train[col].apply(lambda x: int(x[0]))
        test[col+"_"] = test[col].apply(lambda x: int(x[0]))

    train["기존 IVF 성공 확률"] = train.apply(lambda row: row["IVF 출산 횟수_"]/row["IVF 시술 횟수_"] if row["IVF 시술 횟수_"] != 0 else 0, axis = 1)
    train["IVF 누적 성공률 예측"] = train["기존 IVF 성공 확률"].ewm(alpha=0.7).mean()

    test["기존 IVF 성공 확률"] = test.apply(lambda row: row["IVF 출산 횟수_"]/row["IVF 시술 횟수_"] if row["IVF 시술 횟수_"] != 0 else 0, axis = 1)
    test["IVF 누적 성공률 예측"] = test["기존 IVF 성공 확률"].ewm(alpha=0.7).mean()

    train = train.drop(columns = ["기존 IVF 성공 확률", "IVF 출산 횟수_", "IVF 시술 횟수_"])
    test = test.drop(columns = ["기존 IVF 성공 확률", "IVF 출산 횟수_", "IVF 시술 횟수_"])

    return train, test

def FE_11(train_, test_):
    train = train_.copy()
    test = test_.copy()

    train["배아 생존률"] = train.apply(lambda row: row["해동된 배아 수"]/row["저장된 배아 수"] if row["저장된 배아 수"] != 0 else 0, axis = 1)
    test["배아 생존률"] = test.apply(lambda row: row["해동된 배아 수"]/row["저장된 배아 수"] if row["저장된 배아 수"] != 0 else 0, axis = 1)

    return train, test

def FE_12(train_, test_):
    train = train_.copy()
    test = test_.copy()

    for col in ["DI 시술 횟수", "DI 출산 횟수"]:
        train[col+"_"] = train[col].apply(lambda x: int(x[0]))
        test[col+"_"] = test[col].apply(lambda x: int(x[0]))

    train["기존 DI 성공 확률"] = train.apply(lambda row: row["DI 출산 횟수_"]/row["DI 시술 횟수_"] if row["DI 시술 횟수_"] != 0 else 0, axis = 1)
    test["기존 DI 성공 확률"] = test.apply(lambda row: row["DI 출산 횟수_"]/row["DI 시술 횟수_"] if row["DI 시술 횟수_"] != 0 else 0, axis = 1)

    train = train.drop(columns = ["DI 출산 횟수_", "DI 시술 횟수_"])
    test = test.drop(columns = ["DI 출산 횟수_", "DI 시술 횟수_"])

    return train, test

def FE_13(train_, test_):
    train = train_.copy()
    test = test_.copy()

    for col in ["DI 시술 횟수", "DI 출산 횟수"]:
        train[col+"_"] = train[col].apply(lambda x: int(x[0]))
        test[col+"_"] = test[col].apply(lambda x: int(x[0]))

    train["기존 DI 성공 확률"] = train.apply(lambda row: row["DI 출산 횟수_"]/row["DI 시술 횟수_"] if row["DI 시술 횟수_"] != 0 else 0, axis = 1)
    train["DI 누적 성공률 예측"] = train["기존 DI 성공 확률"].ewm(alpha=0.7).mean()

    test["기존 DI 성공 확률"] = test.apply(lambda row: row["DI 출산 횟수_"]/row["DI 시술 횟수_"] if row["DI 시술 횟수_"] != 0 else 0, axis = 1)
    test["DI 누적 성공률 예측"] = test["기존 DI 성공 확률"].ewm(alpha=0.7).mean()

    train = train.drop(columns = ["기존 DI 성공 확률", "DI 출산 횟수_", "DI 시술 횟수_"])
    test = test.drop(columns = ["기존 DI 성공 확률", "DI 출산 횟수_", "DI 시술 횟수_"])

    return train, test

FE = ["FE_1", "FE_2", "FE_3", "FE_4","FE_5", "FE_6", "FE_7", "FE_8", "FE_9", "FE_10", "FE_11", "FE_12", "FE_13"]
FE_dict = {"FE_1":FE_1, "FE_2":FE_2, "FE_3":FE_3, "FE_4":FE_4, "FE_5":FE_5, "FE_6":FE_6, "FE_7":FE_7, "FE_8":FE_8, "FE_9":FE_9, "FE_10":FE_10, "FE_11":FE_11, "FE_12":FE_12, "FE_13":FE_13}
for d in FE:
    train, test = FE_dict[d](raw_train, raw_test)

FE_combination = [[]]
for i in range(len(FE)):
    FE_combination += list(combinations(FE, i))
# FE_combination = [_ for _ in FE_combination if "FE_1" in _]

print(FE_combination)
print(len(FE_combination))

# save = pd.DataFrame({"mix":[], "score":[]})
save = pd.read_csv("backup_lgbm.csv")
start = len(save)

for idx, i in enumerate(FE_combination[start:]):
    print(f"{idx + start} trying....", i)
    try:
        train, test = raw_train.drop(columns = "ID"), raw_test.drop(columns = "ID")
        for fe in i:
            train, test = FE_dict[fe](train, test)

        cat_features = train.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_features:
            most_frequent = train[col].mode()[0]
            train[col] = train[col].fillna(most_frequent)
            test[col] = test[col].fillna(most_frequent)

            # 레이블 인코딩 추가
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col])

            # test 데이터 처리
            test[col] = test[col].apply(lambda x: x if x in le.classes_ else most_frequent)  # train에 없는 값이면 최빈값으로 대체
            test[col] = le.transform(test[col])

        X = train.drop(columns="임신 성공 여부")
        y = train["임신 성공 여부"]

        params = {
                    "objective": "binary",
                    "metric": "auc",
                    "boosting": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "n_estimators": 2000,
                    "verbosity": -1
                }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        auc_scores = []

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
                print(f"\n[Fold {fold+1}] Training...")

                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[
        lgb.early_stopping(stopping_rounds=300),
    ])

                valid_pred = model.predict_proba(X_valid)[:, 1]
                auc = roc_auc_score(y_valid, valid_pred)
                auc_scores.append(auc)
                print(f"[Fold {fold+1}] AUC: {auc:.4f}")

        print(f"{", ".join(i)}\nMean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

        save.loc[len(save)] = [idx + start, ", ".join(i), f"Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}"]
        save.to_csv("backup_lgbm.csv", index=False)
    except:
        save.loc[len(save)] = [idx + start, ", ".join(i), f"오류남, 다시 해"]
        save.to_csv("backup_lgbm.csv", index=False)
    






    
    

