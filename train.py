# battery_model_comparison_optimized.py
import warnings
warnings.filterwarnings("ignore")

import time
import math
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import matplotlib.pyplot as plt

# ========== Параметры ==========
folder_path = "./data"
required_columns = [
    'Voltage_measured', 'Current_measured', 'Temperature_measured',
    'Current_load', 'Voltage_load', 'Time'
]

random_state = 42
test_size = 0.2
cv_splits = 5
n_iter_search = 25   # можно увеличить (больше — дольше)
n_jobs = -1

# каталоги для сохранения
plots_dir = Path("plots")
models_dir = Path(".")
plots_dir.mkdir(parents=True, exist_ok=True)

# ========== 1) Загрузка и фильтрация CSV ==========
def load_battery_data(folder_path, required_columns):
    files = list(Path(folder_path).glob("*.csv"))
    dfs = []
    good_files = []
    for file in files:
        try:
            df = pd.read_csv(file)
            if all(col in df.columns for col in required_columns):
                dfs.append(df[required_columns])  # отбираем только нужные колонки
                good_files.append(file.name)
                if len(good_files) == 20:      ##                                        ОГРАНИЧЕНИЕ
                    break
            else:
                # можно логировать файлы, которые пропустили
                pass
        except Exception as e:
            print(f"Пропущен {file.name}: {e}")
    if not dfs:
        raise ValueError("Не найдено подходящих CSV файлов в папке.")
    data = pd.concat(dfs, ignore_index=True)
    print(f"Загружено {len(good_files)} файлов: {good_files}")
    print(f"Всего строк: {len(data)}")
    return data

data = load_battery_data(folder_path, required_columns)

# ========== 2) Формирование X, y ==========
data = data.dropna(how="all")  # удалить полностью пустые строки, если есть
data['lifetime'] = data['Time']
feature_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load']
X = data[feature_cols].copy()
y = data['lifetime'].copy()

# ========== 3) Train/test split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# ========== 4) Pipeline (imputer + scaler) ==========
preproc = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # на случай пропусков
    ("scaler", StandardScaler())
])

# ========== 5) Модели и сетки для RandomizedSearch ==========
models_and_params = {
    "RandomForest": {
        "estimator": Pipeline([("preproc", preproc), ("model", RandomForestRegressor(random_state=random_state))]),
        "params": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["auto", "sqrt"]
        }
    },
    "GradientBoosting": {
        "estimator": Pipeline([("preproc", preproc), ("model", GradientBoostingRegressor(random_state=random_state))]),
        "params": {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 6, 10]
        }
    },
    "XGBoost": {
        "estimator": Pipeline([("preproc", preproc), ("model", XGBRegressor(objective='reg:squarederror', random_state=random_state, verbosity=0))]),
        "params": {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 6, 10],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0]
        }
    },
    "LightGBM": {
        "estimator": Pipeline([("preproc", preproc), ("model", LGBMRegressor(random_state=random_state, verbose=-1))]),
        "params": {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__num_leaves": [31, 63, 127],
            "model__subsample": [0.6, 0.8, 1.0]
        }
    },
    "CatBoost": {
        "estimator": Pipeline([("preproc", preproc), ("model", CatBoostRegressor(verbose=0, random_state=random_state))]),
        "params": {
            "model__iterations": [100, 200, 400],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__depth": [4, 6, 10],
            "model__l2_leaf_reg": [1, 3, 5]
        }
    }
}

# ========== 6) Поиск лучших гиперпараметров (RandomizedSearchCV) ==========
cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
results_summary = []

start_time = time.time()
for name, cfg in models_and_params.items():
    print(f"\n--- Подбор для {name} ---")
    estimator = cfg["estimator"]
    param_dist = cfg["params"]
    rnd_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring="neg_mean_absolute_error",  # оптимизируем MAE
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
        return_train_score=False
    )
    rnd_search.fit(X_train, y_train)
    best = rnd_search.best_estimator_
    # Оценка на тестовой выборке
    y_pred = best.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Лучшие параметры ({name}): {rnd_search.best_params_}")
    print(f"{name} -> MAE(test): {mae:.4f}, R²(test): {r2:.4f}")
    results_summary.append({
        "model": name,
        "mae": mae,
        "r2": r2,
        "best_estimator": best,
        "best_params": rnd_search.best_params_
    })

elapsed = time.time() - start_time
print(f"\nВремя подбора: {elapsed/60:.2f} мин.")

# ========== 7) Таблица результатов и выбор лучшей ==========
results_df = pd.DataFrame([{"model": r["model"], "MAE": r["mae"], "R2": r["r2"]} for r in results_summary])
results_df = results_df.sort_values("MAE").reset_index(drop=True)
print("\nРезультаты (по MAE):")
print(results_df)

best_row = results_summary[results_df.index[0]]
best_model_name = results_df.loc[0, "model"]
best_model = next(r["best_estimator"] for r in results_summary if r["model"] == best_model_name)
print(f"\nЛучшая модель по MAE: {best_model_name}")

# ========== 8) Сохранение всех моделей (только если файла нет) ==========
saved_models = {}
for r in results_summary:
    model_name = r["model"]
    estimator = r["best_estimator"]
    filename = models_dir / f"model_{model_name}.joblib"
    if not filename.exists():
        joblib.dump(estimator, filename)
        print(f"Сохранено: {filename}")
    else:
        print(f"Файл {filename} уже существует — пропускаем сохранение.")
    saved_models[model_name] = str(filename)

# ========== 9) Визуализация для всех моделей (от лучшей к худшей) ==========
# сортируем по MAE (меньшее -> лучше)
sorted_results = sorted(results_summary, key=lambda r: r['mae'])
n = len(sorted_results)
ncols = 2 if n <= 4 else 3
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows))
axes = np.array(axes).reshape(-1)  # flatten (works even если 1D)
for idx, res in enumerate(sorted_results):
    model_name = res['model']
    estimator = res['best_estimator']
    y_pred_model = estimator.predict(X_test)

    ax = axes[idx]
    ax.scatter(y_test, y_pred_model, alpha=0.5)
    mn, mx = min(y_test.min(), y_pred_model.min()), max(y_test.max(), y_pred_model.max())
    ax.plot([mn, mx], [mn, mx], linestyle='--', color='red')
    ax.set_title(f"{model_name} (MAE={res['mae']:.2f}, R2={res['r2']:.4f})")
    ax.set_xlabel("Фактическое")
    ax.set_ylabel("Предсказанное")

    # сохраняем отдельный файл для каждой модели
    fig_ind = plt.figure(figsize=(6,4))
    ax_ind = fig_ind.add_subplot(1,1,1)
    ax_ind.scatter(y_test, y_pred_model, alpha=0.5)
    ax_ind.plot([mn, mx], [mn, mx], linestyle='--', color='red')
    ax_ind.set_title(f"{model_name} (MAE={res['mae']:.2f}, R2={res['r2']:.4f})")
    ax_ind.set_xlabel("Фактическое")
    ax_ind.set_ylabel("Предсказанное")
    ind_path = plots_dir / f"plot_{model_name}.png"
    fig_ind.tight_layout()
    fig_ind.savefig(ind_path)
    plt.close(fig_ind)
    print(f"Сохранён график для {model_name}: {ind_path}")

# выключаем пустые оси (если есть)
total_axes = nrows * ncols
for j in range(n, total_axes):
    axes[j].axis('off')

# сохраняем общий график
combined_path = plots_dir / f"plots_all_sorted_by_MAE.png"
fig.tight_layout()
fig.savefig(combined_path)
plt.close(fig)
print(f"Сохранён комбинированный график: {combined_path}")

# ========== 10) Предсказание на вводе пользователя (пример) ==========
try:
    vm = float(input("Введите Напряжение (Voltage_measured): "))
    c = float(input("Введите Ток (Current_measured): "))
    t = float(input("Введите Температуру (Temperature_measured): "))
    l = float(input("Введите Ток нагрузки (Current_load): "))
    vl = float(input("Введите Нагруженное Напряжение (Voltage_load): "))
    new_data = pd.DataFrame({
        'Voltage_measured': [vm],
        'Current_measured': [c],
        'Temperature_measured': [t],
        'Current_load': [l],
        'Voltage_load': [vl]
    })

    print("\nПредсказания всех моделей:")
    max_lifetime = data['Time'].max()
    print(f"######## MAX ВРЕМЯ ЖИЗНИ : {max_lifetime} #############")
    for model_name, filename in saved_models.items():
        model = joblib.load(filename)
        pred = model.predict(new_data)[0]
        remaining_capacity_percent = (1 - (pred / max_lifetime)) * 100
        print(f"{model_name}: Остаток емкости: {remaining_capacity_percent:.2f}%")
except Exception:
    print("Ввод пропущен или некорректен — предсказание пропущено.")
