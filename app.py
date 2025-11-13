# 0. 라이브러리

import streamlit as st
import pandas as pd
import numpy as np
import io 
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.1)


# 1. 공통 함수

def get_metrics(model, X_ev, y_ev):
    """모델과 평가 데이터를 받아 지표 딕셔너리를 반환"""
    try:
        proba = model.predict_proba(X_ev)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(y_ev, pred),
            "precision": precision_score(y_ev, pred, zero_division=0),
            "recall": recall_score(y_ev, pred, zero_division=0),
            "f1": f1_score(y_ev, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_ev, proba),
        }
    except Exception as e:
        st.error(f"지표 계산 중 오류: {e}")
        return None

def plot_confusion(y_true, y_pred, cmap="Blues"):
    """Confusion Matrix 차트 생성"""
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, proba, name):
    """ROC Curve 차트 생성"""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("ROC Curve")
    plt.tight_layout()
    return fig

@st.cache_data
def convert_fig_to_png(_fig): # 1. fig -> _fig 로 변경 (캐시 오류 수정)
    """Matplotlib Figure를 PNG 이미지 바이트로 변환 (캐시 해시 비활성화)"""
    buf = io.BytesIO()
    _fig.savefig(buf, format="png", bbox_inches='tight') # 2. 내부 변수도 _fig 로 변경
    buf.seek(0)
    return buf.getvalue()

@st.cache_data
def convert_df_to_csv(df):
    """DataFrame을 CSV 바이트로 변환"""
    return df.to_csv(index=True).encode('utf-8')


# 2. 데이터 분할 및 모델 훈련 함수 (캐시 없음)

def split_data(X, y, test_ratio, val_ratio):
    """사용자 설정 비율에 따라 Train/Val/Test로 분할"""
    # 1단계: Test 분리
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=test_ratio, 
        stratify=y, 
        random_state=42
    )
    
    # 2단계: 남은 데이터(Train+Val)에서 Val 분리
    # (Train+Val) 크기 대비 Val 크기 비율 계산
    if (1.0 - test_ratio) == 0: # test_ratio가 1.0일 경우 방지
        val_ratio_within = 0
    else:
        val_ratio_within = val_ratio / (1.0 - test_ratio)
    
    # val_ratio_within이 1.0 이상이면 Val이 Train+Val보다 크게 설정된 것이므로 조정
    if val_ratio_within >= 1.0:
        val_ratio_within = 0.99 # 거의 모든 것을 Val로 (비정상적이지만 에러 방지)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio_within,
        stratify=y_train_val,
        random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_models(X_train, y_train, numeric_features, categorical_features):
    """전처리 파이프라인을 포함한 4개 모델을 훈련"""
    
    # transformers 리스트를 동적으로 구성합니다. (ValueError 수정)
    transformers_list = []

    if numeric_features: # 수치형 변수가 하나라도 있을 때만 추가
        transformers_list.append(
            ("num", StandardScaler(), numeric_features)
        )
    
    if categorical_features: # 범주형 변수가 하나라도 있을 때만 추가
        transformers_list.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        )

    if not transformers_list:
        st.warning("분석할 수치형 또는 범주형 변수가 선택되지 않았습니다.")

    # 전처리 파이프라인
    preprocess = ColumnTransformer(
        transformers=transformers_list, # 동적으로 생성된 리스트 사용
        remainder="passthrough" # 선택되지 않은 피처는 통과시킴
    )

    # 5-fold 교차 검증 설정
    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models_dict = {}
    
    with st.spinner("1/4: Logistic Regression 훈련 중..."):
        log_pipeline = Pipeline(steps=[
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
        ])
        log_param_grid = {"clf__C": [0.1, 1.0, 10.0]} # 속도를 위해 파라미터 축소
        log_cv = GridSearchCV(log_pipeline, log_param_grid, scoring="roc_auc", cv=cv_stratified, n_jobs=-1, refit=True)
        log_cv.fit(X_train, y_train)
        models_dict["Logistic"] = log_cv.best_estimator_

    with st.spinner("2/4: Decision Tree 훈련 중..."):
        dt_pipeline = Pipeline(steps=[
            ("preprocess", preprocess),
            ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42))
        ])
        dt_param_grid = {"clf__max_depth": [5, 10, None], "clf__min_samples_split": [2, 10]}
        dt_cv = GridSearchCV(dt_pipeline, dt_param_grid, scoring="roc_auc", cv=cv_stratified, n_jobs=-1, refit=True)
        dt_cv.fit(X_train, y_train)
        models_dict["Decision Tree"] = dt_cv.best_estimator_

    with st.spinner("3/4: Random Forest 훈련 중..."):
        rf_pipeline = Pipeline(steps=[
            ("preprocess", preprocess),
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))
        ])
        rf_param_grid = {"clf__n_estimators": [100, 200], "clf__max_depth": [10, None]}
        rf_cv = GridSearchCV(rf_pipeline, rf_param_grid, scoring="roc_auc", cv=cv_stratified, n_jobs=-1, refit=True)
        rf_cv.fit(X_train, y_train)
        models_dict["Random Forest"] = rf_cv.best_estimator_

    if XGB_AVAILABLE:
        with st.spinner("4/4: XGBoost 훈련 중..."):
            # 클래스 불균형 비율 계산
            pos = y_train.sum()
            neg = (y_train == 0).sum()
            scale_pos_weight = neg / pos if pos > 0 else 1 # 0으로 나누는 오류 방지
            
            xgb_pipeline = Pipeline(steps=[
                ("preprocess", preprocess),
                ("clf", XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss", scale_pos_weight=scale_pos_weight))
            ])
            xgb_param_grid = {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5]}
            xgb_cv = GridSearchCV(xgb_pipeline, xgb_param_grid, scoring="roc_auc", cv=cv_stratified, n_jobs=-1, refit=True)
            xgb_cv.fit(X_train, y_train)
            models_dict["XGBoost"] = xgb_cv.best_estimator_
    else:
        st.warning("XGBoost 라이브러리가 설치되지 않았습니다. XGBoost 모델을 제외하고 분석합니다.")
    
    return models_dict


# 3. Streamlit 앱 메인 함수

def main():
    st.set_page_config(page_title="범용 분류 모델 비교 대시보드", layout="wide")
    st.title("👍 범용 분류 모델 비교 대시보드")
    st.markdown("어떤 CSV 파일이든 업로드하여 4가지 주요 분류 모델의 성능을 비교, 평가, 시각화합니다.")

    # --- Session State 초기화 ---
    # (앱 재실행 시 유지되어야 할 값들)
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False # 분석 실행 여부
    if 'metrics_df' not in st.session_state:
        st.session_state.metrics_df = None # 평가 결과
    if 'models_dict' not in st.session_state:
        st.session_state.models_dict = None # 훈련된 모델
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = None # 타깃 인코더
    if 'test_data' not in st.session_state:
        st.session_state.test_data = (None, None) # (X_test, y_test)
    if 'final_metric' not in st.session_state:
        st.session_state.final_metric = 'recall' # 최종 선택 지표
    if 'sample_loaded' not in st.session_state:
        st.session_state.sample_loaded = False
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None


    # --- 사이드바 설정 ---
    st.sidebar.header("⚙️ 1. 분석 설정")
    uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

    if uploaded_file is None:
        st.info("👈 사이드바에서 CSV 파일을 업로드하여 분석을 시작하세요.")
        st.markdown("")
        # 샘플 데이터 사용 옵션 (cleaned_hr_attrition_dataset.csv가 있다고 가정)
        if os.path.exists("cleaned_hr_attrition_dataset.csv"):
            st.sidebar.markdown("---")
            if st.sidebar.button("샘플 HR 데이터로 분석하기"):
                uploaded_file = "cleaned_hr_attrition_dataset.csv"
                st.session_state.sample_loaded = True # 샘플 로드 상태 저장
                st.session_state.analysis_run = False # 샘플 로드시 분석 상태 초기화
                st.rerun() # 페이지 새로고침
        else:
             st.sidebar.caption("'cleaned_hr_attrition_dataset.csv' 파일을 같은 폴더에 두면 샘플 분석을 제공합니다.")
    
    # (세션 상태를 이용해 샘플 로드 유지)
    if not uploaded_file and st.session_state.sample_loaded:
         uploaded_file = "cleaned_hr_attrition_dataset.csv"

    if uploaded_file is not None:
        try:
            # 파일이 변경되면 분석 상태 초기화
            file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file
            if st.session_state.current_file != file_name:
                st.session_state.analysis_run = False
                st.session_state.current_file = file_name

            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"'{file_name}' 로드 성공!")
            
            with st.expander("데이터 미리보기 (상위 5행)"):
                st.dataframe(df.head())

            # --- 2. 변수 설정 ---
            st.sidebar.header("🎯 2. 변수 설정")
            all_columns = df.columns.tolist()
            
            # (샘플 데이터의 경우 기본값 설정)
            default_target_idx = 0
            if "AttritionFlag" in all_columns:
                default_target_idx = all_columns.index("AttritionFlag")
            elif len(all_columns) > 0:
                default_target_idx = len(all_columns) - 1 # 마지막 컬럼
                
            target_variable = st.sidebar.selectbox(
                "타깃(Y) 변수 선택 (필수, 2개 값)",
                all_columns,
                index=default_target_idx
            )
            
            feature_candidates = [col for col in all_columns if col != target_variable]
            excluded_features = st.sidebar.multiselect(
                "분석에서 제외할 변수 선택",
                feature_candidates,
                default=[]
            )
            selected_features = [col for col in feature_candidates if col not in excluded_features]

            # --- 3. 데이터 분할 설정 ---
            st.sidebar.header("✂️ 3. 데이터 분할 비율")
            test_ratio = st.sidebar.slider("테스트(Test) 세트 비율", 0.1, 0.5, 0.15, 0.05)
            val_ratio = st.sidebar.slider("검증(Validation) 세트 비율", 0.1, 0.5, 0.25, 0.05)
            
            train_ratio = 1.0 - test_ratio - val_ratio
            
            if train_ratio <= 0.1: # 훈련셋이 너무 작으면 경고
                st.sidebar.error(f"훈련 세트 비율이 {train_ratio*100:.0f}%로 너무 낮습니다. 테스트/검증 비율을 낮춰주세요.")
                st.stop()
            else:
                st.sidebar.info(f"훈련 세트 비율: **{train_ratio*100:.0f}%**")

            # --- 4. 분석 실행 버튼 ---
            st.sidebar.markdown("---")
            if st.sidebar.button("🚀 모델 훈련 및 분석 시작", type="primary", use_container_width=True):
                
                # --- [시작] 분석 파이프라인 ---
                st.header("🔬 1. 분석 준비")
                
                y = df[target_variable]
                if y.nunique() != 2:
                    st.error(f"오류: 타깃 변수 '{target_variable}'의 고유값이 2개가 아닙니다 (현재: {y.nunique()}개). 이진 분류만 지원합니다.")
                    st.stop()
                
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                target_mapping = {label: idx for idx, label in enumerate(le.classes_)}
                st.info(f"타깃 변수 '{target_variable}' 인코딩: {target_mapping}")

                X = df[selected_features]
                numeric_features = X.select_dtypes(include=np.number).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not numeric_features and not categorical_features:
                    st.error("오류: 분석할 피처(X) 변수가 없습니다. '제외할 변수' 설정을 확인하세요.")
                    st.stop()

                st.write(f"**총 {len(selected_features)}개 피처 사용:**")
                st.write(f"- 📈 **수치형({len(numeric_features)}개):** `{', '.join(numeric_features) if numeric_features else '없음'}`")
                st.write(f"- 🔠 **범주형({len(categorical_features)}개):** `{', '.join(categorical_features) if categorical_features else '없음'}`")

                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded, test_ratio, val_ratio)
                st.write(f"**데이터 분할 결과:** 훈련 {len(y_train)}개, 검증 {len(y_val)}개, 테스트 {len(y_test)}개")
                
                st.header("🏃 2. 모델 훈련 및 평가")
                models_dict = train_models(X_train, y_train, numeric_features, categorical_features)

                rows = []
                for name, model in models_dict.items():
                    val_metrics = get_metrics(model, X_val, y_val)
                    test_metrics = get_metrics(model, X_test, y_test)
                    if val_metrics: rows.append({"model": name, "set": "Validation", **val_metrics})
                    if test_metrics: rows.append({"model": name, "set": "Test", **test_metrics})
                
                metrics_df = pd.DataFrame(rows).set_index(["model", "set"]).round(4)
                
                # --- Session State에 결과 저장 ---
                st.session_state.analysis_run = True
                st.session_state.metrics_df = metrics_df
                st.session_state.models_dict = models_dict
                st.session_state.label_encoder = le
                st.session_state.test_data = (X_test, y_test)
                st.rerun() # 버튼 클릭 후 즉시 재실행하여 아래 `if` 블록을 타도록 함

        except pd.errors.ParserError:
            st.error("오류: CSV 파일을 읽는 데 실패했습니다. 파일이 손상되었거나 유효한 CSV 형식이 아닌지 확인하세요.")
            st.session_state.analysis_run = False
        except KeyError as e:
            st.error(f"오류: '{e}' 컬럼을 찾을 수 없습니다. 사이드바에서 변수 설정을 다시 확인하세요.")
            st.session_state.analysis_run = False
        except Exception as e:
            st.error(f"분석 준비 중 예기치 않은 오류가 발생했습니다: {e}")
            st.session_state.analysis_run = False

    # --- 분석 결과 표시 로직 (Session State 기반) ---
    if st.session_state.analysis_run:
        # Session State에서 결과 불러오기
        metrics_df = st.session_state.metrics_df
        models_dict = st.session_state.models_dict
        le = st.session_state.label_encoder
        X_test, y_test = st.session_state.test_data

        if metrics_df is None or models_dict is None or le is None or X_test is None:
             st.warning("분석 결과가 없습니다. 사이드바에서 '모델 훈련 및 분석 시작' 버튼을 눌러주세요.")
             st.stop()
             
        # --- 1.5. 모델 평가 (표시) ---
        st.header("📊 2. 모델 성능 비교표")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="lightgreen"))
        st.download_button(
            label="성능 비교표 (CSV) 다운로드",
            data=convert_df_to_csv(metrics_df),
            file_name="model_metrics.csv",
            mime="text/csv",
        )

        # --- 1.6. 성능 시각화 (표시) ---
        st.header("📈 3. 성능 시각화 (Test Set 기준)")
        test_metrics = metrics_df.xs("Test", level="set")
        
        col1, col2 = st.columns(2)
        
        # ROC-AUC 비교
        fig_roc_comp = plt.figure(figsize=(7, 5))
        test_metrics["roc_auc"].sort_values().plot(kind="barh", ax=fig_roc_comp.add_subplot(111))
        plt.title("Test Set: ROC-AUC Comparison")
        plt.xlabel("ROC-AUC Score")
        col1.pyplot(fig_roc_comp)
        
        # Recall 비교
        fig_recall_comp = plt.figure(figsize=(7, 5))
        test_metrics["recall"].sort_values().plot(kind="barh", ax=fig_recall_comp.add_subplot(111))
        plt.title("Test Set: Recall Comparison")
        plt.xlabel("Recall Score")
        col2.pyplot(fig_recall_comp)

        st.download_button(
            label="ROC-AUC 비교 차트 (PNG) 다운로드",
            data=convert_fig_to_png(fig_roc_comp),
            file_name="roc_auc_comparison.png",
            mime="image/png",
        )

        # --- 1.7. 모델별 상세 분석 (표시) ---
        st.header("🔍 4. 모델별 상세 분석 (Test Set)")
        tab_names = list(models_dict.keys())
        tabs = st.tabs(tab_names)
        
        for i, name in enumerate(tab_names):
            with tabs[i]:
                model = models_dict[name]
                proba = model.predict_proba(X_test)[:, 1]
                pred = (proba >= 0.5).astype(int)
                
                st.subheader(f"{name}: 최적 하이퍼파라미터")
                st.json(model.named_steps['clf'].get_params())
                
                tcol1, tcol2 = st.columns([1, 2])
                
                fig_cm = plot_confusion(y_test, pred, cmap="Reds" if name == "Logistic" else "Blues")
                tcol1.pyplot(fig_cm)
                
                fig_roc_ind = plot_roc_curve(y_test, proba, name)
                tcol2.pyplot(fig_roc_ind)
                
                # --- [수정] Classification Report를 DataFrame으로 변환 ---
                st.subheader("Classification Report")
                try:
                    # output_dict=True로 딕셔너리 받기
                    report_dict = classification_report(y_test, pred, target_names=[str(c) for c in le.classes_], output_dict=True)
                    # DataFrame으로 변환
                    report_df = pd.DataFrame(report_dict).transpose().round(4)
                    # st.dataframe으로 깔끔하게 표시
                    st.dataframe(report_df)
                except Exception as e:
                    st.error(f"Report 생성 중 오류: {e}")
                    st.text(classification_report(y_test, pred, target_names=[str(c) for c in le.classes_])) # 실패 시 텍스트로 표시
                # --- [수정 끝] ---

        # --- 1.8. 최종 결론 (표시) ---
        st.header("💡 5. 최종 결론")
        st.subheader("👌핵심 지표에 따른 최적 모델")
        
        metric_to_optimize = st.selectbox(
            "비즈니스 목표에 가장 중요한 핵심 지표(Metric)를 선택하세요:",
            ["recall", "roc_auc", "accuracy", "precision", "f1"],
            key='final_metric' # st.session_state와 연동
        )
        
        # test_metrics가 비어있지 않은지 확인
        if not test_metrics.empty:
            best_model_name = test_metrics[st.session_state.final_metric].idxmax()
            best_score = test_metrics.loc[best_model_name, st.session_state.final_metric]
            
            st.success(f"**'{st.session_state.final_metric.upper()}'** 지표 기준, 최적 모델은 **'{best_model_name}'** (점수: {best_score:.4f}) 입니다.")
        else:
            st.warning("Test Set 평가지표를 계산할 수 없습니다.")
        
        # --- [수정] 지표 설명을 HR 예시로 변경 및 Accuracy 추가 ---
        st.markdown(
            """
            - **Accuracy (정확도)가 중요하다면?**
                - **(예시: HR 분석)** 전체 직원 중 '이직자'와 '잔류자'를 모두 얼마나 정확하게 예측했는지가 중요할 때 선택합니다.
                - **(주의)** 만약 잔류자가 95%고 이직자가 5%라면, 모델이 전부 '잔류'로 예측해도 정확도는 95%가 나옵니다. 데이터가 불균형할 땐 신뢰하기 어려운 지표입니다.

            - **Recall (재현율)이 중요하다면?**
                - **(예시: HR 분석)** 실제 이직할 직원(Positive)을 놓치지 않고 찾아내는 것이 목표일 때 선택합니다. (예: 핵심 인재 유출 방지)
                - **False Negative (FN) 비용**이 매우 클 때 (예: 이직할 핵심 인재를 '잔류'로 잘못 예측하여 아무 조치도 못 하고 놓침) 이 지표를 높여야 합니다.

            - **Precision (정밀도)이 중요하다면?**
                - **(예시: HR 분석)** 모델이 **'이직자(Positive)'라고 예측한 사람**이 실제로 이직할 확률이 높아야 할 때 선택합니다.
                - **False Positive (FP) 비용**이 매우 클 때 (예: 잔류할 직원을 '이직자'로 잘못 예측하여 불필요한 면담, 보너스 지급 등 리소스를 낭비함) 이 지표를 높여야 합니다.

            - **ROC-AUC가 중요하다면?**
                - 모델이 '이직자'와 '잔류자'를 얼마나 잘 **구별**하는지 나타내는 전반적인 성능 지표입니다.
                - Recall과 Precision이 상충(Trade-off) 관계일 때, 모델의 종합적인 분류 성능을 판단하기 좋습니다.
            
            - **F1-Score가 중요하다면?**
                - Precision과 Recall의 **조화 평균**입니다. 두 지표가 모두 중요하지만 데이터가 불균형할 때 (예: 이직자가 5%인 경우) Accuracy보다 신뢰할 수 있습니다.
            """
        )
        st.balloons()


# 4. 스크립트 실행

if __name__ == "__main__":
    main()

# 4. 스크립트 실행

if __name__ == "__main__":
    main()
