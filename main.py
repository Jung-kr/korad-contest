# 현재 청구된 전기세를 입력하면 원자력발전으로 생산된 전기가 화력발전으로 생산된 전기로 대체되었을 때의 오른 전기세를 예측하여 출력해주는 모델
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression, Ridge
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()



# 전역 변수로 학습된 모델 및 데이터 비율 저장
lrp = None
industry_to_housing = None
general_to_housing = None

def load_and_preprocess_data():
    global lrp, industry_to_housing, general_to_housing
    # 데이터 로드 및 전처리
    seoul_1 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (1).xls", header = 11)
    seoul_2 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (2).xls", header = 11)
    seoul_3 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (3).xls", header = 11)
    seoul_4 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (4).xls", header = 11)
    seoul_5 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (5).xls", header = 11)
    seoul_6 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (6).xls", header = 11)
    seoul_7 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (7).xls", header = 11)
    seoul_8 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (8).xls", header = 11)
    seoul_9 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (9).xls", header = 11)
    seoul_10 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (10).xls", header = 11) 
    seoul_11 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (11).xls", header = 11)

    gyeonggi_1 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (12).xls", header = 11)
    gyeonggi_2 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (13).xls", header = 11)
    gyeonggi_3 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (14).xls", header = 11)
    gyeonggi_4 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (15).xls", header = 11)
    gyeonggi_5 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (16).xls", header = 11)
    gyeonggi_6 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (17).xls", header = 11)
    gyeonggi_7 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (18).xls", header = 11)
    gyeonggi_8 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (19).xls", header = 11)
    gyeonggi_9 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (20).xls", header = 11)
    gyeonggi_10 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (21).xls", header = 11)
    gyeonggi_11 = pd.read_excel("/app/가구별_전력사용량_data/가구 평균 월별 전력사용량_20241001 (22).xls", header = 11)

    df_seoul = pd.concat([seoul_1, seoul_2, seoul_3, seoul_4, seoul_5, seoul_6, seoul_7, seoul_8, seoul_9, seoul_10, seoul_11])
    df_gyeonggi = pd.concat([gyeonggi_1, gyeonggi_2, gyeonggi_3, gyeonggi_4, gyeonggi_5, gyeonggi_6, gyeonggi_7, gyeonggi_8, gyeonggi_9, gyeonggi_10, gyeonggi_11])

    electric = pd.concat([df_seoul, df_gyeonggi])
    electric['대상가구수(호)'] = electric['대상가구수(호)'].str.replace(',', '').str.strip().astype(int)
    electric['가구당 평균 전기요금(원)'] = electric['가구당 평균 전기요금(원)'].str.replace(',', '').str.strip().astype(int)

    # int type으로 되어있는 '년월' 컬럼을 날짜 타입으로 변환 
    electric['년월'] = pd.to_datetime(electric['년월'], format='%Y%m')

    # 예측을 위한 데이터 구성 및 전처리
    mean_elec = electric.groupby('년월')[['대상가구수(호)', '가구당 평균 전력 사용량(kWh)','가구당 평균 전기요금(원)']].mean()

    # 모델 학습용 데이터 준비
    elec_price = pd.read_excel("/app/발전원별_전력거래_정산단가.xlsx")
    elec_trade_rate = pd.read_excel("/app/발전원별_전력거래량_비율.xlsx", header = 68)

    # object type 으로 되어있는 '기간' 컬럼을 날짜타입으로 변환
    elec_price['기간'] = pd.to_datetime(elec_price['기간'])

    # object type 으로 되어있는 '기간' 컬럼을 날짜타입으로 변환
    elec_trade_rate['기간'] = pd.to_datetime(elec_trade_rate['기간'])

    # 총 합 2가 나와야 하는데 화력발전 컬럼으로인해 중볻되는 값이 있음
    elec_trade_rate.iloc[:, 1:].sum(axis = 1)

    # electric 데이터프레임에 있는 평균 전력생산량 컬럼과 ekec_trade_rate 의 발전원별 전력거래 비율을 곱하기 위해 elec_trade의 중복되는 컬럼을 삭제
    elec_trade_rate = elec_trade_rate.drop(['유연탄', '무연탄', 'LNG'], axis = 1)
    elec_trade_rate.iloc[:, 1:].sum(axis = 1)

    # 데이터 병합 후 필요없는 열 삭제
    merged_df = pd.merge(mean_elec, elec_trade_rate, left_on='년월', right_on='기간', how='inner')
    merged3_df = pd.merge(elec_price, merged_df, left_on='기간', right_on='기간', how='inner')

    main_df = merged3_df.loc[:, ['가구당 평균 전기요금(원)']].copy()

    # 열 추가
    main_df['y'] = merged3_df['원자력_x'] * merged3_df['원자력_y'] * merged3_df['가구당 평균 전력 사용량(kWh)'] + merged3_df['화력발전(유연탄+무연탄+LNG)_x'] * merged3_df['원자력_y'] * merged3_df['가구당 평균 전력 사용량(kWh)']
    main_df['y^'] = (
        merged3_df['화력발전(유연탄+무연탄+LNG)_x'] * (merged3_df['원자력_y'] + merged3_df['화력발전(유연탄+무연탄+LNG)_y'])  * merged3_df['가구당 평균 전력 사용량(kWh)'])
    main_df['합계(원/kWh)'] = merged3_df['합계(원/kWh)']
    main_df['가구당 평균 전력 사용량(kWh)'] = merged3_df['가구당 평균 전력 사용량(kWh)']

    # 학습 데이터 준비
    x = main_df.drop(["가구당 평균 전기요금(원)", "y^"], axis = 1)
    y = main_df["가구당 평균 전기요금(원)"]

    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.25, random_state = 11)

    # 모델 학습
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # "가구당 평균 전기요금(원)" = -2980.997 + 0.211 * y - 2,456 * 합계(원/kWh) + 108.933 * 가구당 평균 전력 사용량(kWh)
    intercept = lr.intercept_
    coefficients = lr.coef_

    # 'y^' 컬럼의 값을 가져오기
    y_hat_values = main_df['y^']

    # 각 행별로 가구당 평균 전기요금 계산
    predicted_costs = (
        intercept +
        coefficients[0] * y_hat_values +  # y^ 값 사용
        coefficients[1] * main_df['합계(원/kWh)'] +  # 합계(원/kWh)
        coefficients[2] * main_df['가구당 평균 전력 사용량(kWh)']  # 가구당 평균 전력 사용량
    )

    # 결과를 main_df에 추가
    main_df['predicted_average_cost_with_y_hat'] = predicted_costs

    intercept = lr.intercept_
    coefficients = lr.coef_

    # 각 행별로 가구당 평균 전기요금 계산
    pred_origin_costs = (
        intercept +
        coefficients[0] * main_df["y"] +  # y 값 사용
        coefficients[1] * main_df['합계(원/kWh)'] +  # 합계(원/kWh)
        coefficients[2] * main_df['가구당 평균 전력 사용량(kWh)']  # 가구당 평균 전력 사용량
    )

    # 결과를 main_df에 추가
    main_df['pred_y'] = pred_origin_costs

    df = main_df.loc[:, ["pred_y", "predicted_average_cost_with_y_hat"]]

    x = df.drop(["predicted_average_cost_with_y_hat"], axis = 1)
    y = main_df["predicted_average_cost_with_y_hat"]

    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.25, random_state = 10)

    lrp = LinearRegression()
    lrp.fit(x_train, y_train)

    # y^ = -8266.644 + 1.7751 * y

    # 계약종별 판매단가 데이터 로드
    price_contract = pd.read_excel("/app/연간_계약종별 _판매단가.xlsx")
    price_contract.tail()

    # 2024년 데이터 필터링
    filtered_data = price_contract[price_contract['연도'] == 2024.0]

    # 주택용을 분모로 하는 비율 계산
    industry_to_housing = filtered_data['산업용'].values[0] / filtered_data['주택용'].values[0]
    general_to_housing = filtered_data['일반용'].values[0] / filtered_data['주택용'].values[0]

    # 결과 출력
    print(f"산업용 / 주택용 비율: {industry_to_housing}")
    print(f"일반용 / 주택용 비율: {general_to_housing}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_and_preprocess_data()
    yield  # 이 부분은 서버가 종료될 때까지 유지됨

app = FastAPI(lifespan=lifespan)    

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서의 요청을 허용하려면 ["*"]로 설정
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드를 허용
    allow_headers=["*"],  # 모든 헤더를 허용
)

# 예측 함수 정의
def predict_cost(pred_y_value, usage_type):
    # 기본 예측값 계산
    input_data = pd.DataFrame({'pred_y': [pred_y_value]})
    predicted_cost = lrp.predict(input_data)[0]
    
    # 용도에 따라 예측값 조정
    if usage_type == '산업용':
        predicted_cost *= industry_to_housing
    elif usage_type == '일반용':
        predicted_cost *= general_to_housing
    # 주택용은 예측값 그대로 사용
    
    return predicted_cost

# 쿼리 파라미터로 입력받아 예측값 반환하는 API
@app.get("/predict/")
def predict(pred_y_input: float, usage_type_input: str):
    # 입력값 소문자로 변환
    usage_type = usage_type_input.lower()  
    predicted_value = predict_cost(pred_y_input, usage_type)
    return {"predicted_value": predicted_value}