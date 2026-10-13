import csv
import io
import os
import re
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from scipy.interpolate import lagrange, CubicSpline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

APP_TITLE = "TROPI SNACKS S.A.C. | Estimación de deshidratación"

def read_csv_flexible(path):
    errors = []
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, header=None)
        except Exception as e:
            errors.append(str(e))
    raise ValueError("No se pudo leer el CSV: " + "; ".join(errors))

def clean_data(raw):
    df = raw.dropna(how="all", axis=1).copy()
    header_idx = None
    for i in range(min(8, len(df))):
        row = df.iloc[i].astype(str).str.strip()
        joined = " ".join(row.fillna("").astype(str).tolist()).lower()
        if "tiempo" in joined and "humedad" in joined and ("masa" in joined or "perdida" in joined or "pérdida" in joined):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("No encontré una fila de encabezados con Tiempo, Humedad y Perdida_masa.")
    df.columns = [str(v).replace("\ufeff", "").strip() for v in df.iloc[header_idx]]
    df = df.iloc[header_idx + 1:].copy()
    mapped = {}
    for col in df.columns:
        raw_name = str(col).strip().lower()
        key = re.sub(r"[^a-záéíóúñ]", "", raw_name)
        if "tiempo" in key or key == "time":
            mapped[col] = "Tiempo"
        elif "humedad" in key or "moisture" in key:
            mapped[col] = "Humedad"
        elif "perdida" in key or "pérdida" in raw_name or "loss" in key:
            mapped[col] = "Perdida_masa"
    df = df.rename(columns=mapped)
    needed = ["Tiempo", "Humedad", "Perdida_masa"]
    missing = [x for x in needed if x not in df.columns]
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(missing))
    for c in needed:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df = df[needed].dropna().sort_values("Tiempo")
    df = df.groupby("Tiempo", as_index=False)[["Humedad", "Perdida_masa"]].mean()
    if len(df) < 3:
        raise ValueError("Se requieren al menos 3 tiempos diferentes para comparar Lagrange, Spline y ANN.")
    return df.reset_index(drop=True)

def parse_times(s):
    try:
        vals = [float(v.strip()) for v in s.split(",") if v.strip()]
        if not vals or not np.all(np.isfinite(vals)):
            raise ValueError
        return np.array(vals, dtype=float)
    except Exception:
        raise ValueError("Escribe tiempos numéricos separados por comas, por ejemplo: 12, 20, 28.")

def ann_predict(x, y, q):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        raise ValueError("La ANN necesita al menos 3 datos válidos.")
    scaler = StandardScaler()
    xs = scaler.fit_transform(x.reshape(-1, 1))
    qs = scaler.transform(q.reshape(-1, 1))
    solver = "lbfgs" if len(x) < 500 else "adam"
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation="tanh",
                         solver=solver, alpha=0.001, max_iter=8000, random_state=42)
    model.fit(xs, y)
    return model.predict(qs)

def metric_row(equipo, variable, modelo, real, pred):
    mask = np.isfinite(real) & np.isfinite(pred)
    real, pred = np.asarray(real)[mask], np.asarray(pred)[mask]
    if len(real) < 2:
        return {"Equipo": equipo, "Variable": variable, "Modelo": modelo, "R²": np.nan, "MAE": np.nan, "RMSE": np.nan}
    return {"Equipo": equipo, "Variable": variable, "Modelo": modelo,
            "R²": round(float(r2_score(real, pred)), 4),
            "MAE": round(float(mean_absolute_error(real, pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(real, pred))), 4)}

def calculate(df, times, equipment):
    x = df["Tiempo"].to_numpy()
    h = df["Humedad"].to_numpy()
    p = df["Perdida_masa"].to_numpy()
    lh, lp = lagrange(x, h), lagrange(x, p)
    sh, sp = CubicSpline(x, h, bc_type="natural"), CubicSpline(x, p, bc_type="natural")
    ah, ap = ann_predict(x, h, times), ann_predict(x, p, times)
    result = pd.DataFrame({
        "Equipo": equipment,
        "Tiempo (min)": times,
        "Humedad - Lagrange (%)": np.round(lh(times), 4),
        "Humedad - Spline cúbico (%)": np.round(sh(times), 4),
        "Humedad - ANN (%)": np.round(ah, 4),
        "Pérdida masa - Lagrange (%)": np.round(lp(times), 4),
        "Pérdida masa - Spline cúbico (%)": np.round(sp(times), 4),
        "Pérdida masa - ANN (%)": np.round(ap, 4),
    })
    metricas = []
    for name, ph, pp in [
        ("Lagrange", lh(x), lp(x)),
        ("Spline cúbico", sh(x), sp(x)),
        ("Red neuronal (ANN)", ann_predict(x, h, x), ann_predict(x, p, x))
    ]:
        metricas.append(metric_row(equipment, "Humedad", name, h, ph))
        metricas.append(metric_row(equipment, "Pérdida de masa", name, p, pp))
    models = {"x": x, "h": h, "p": p, "lh": lh, "lp": lp, "sh": sh, "sp": sp}
    return result, pd.DataFrame(metricas), models

class TropiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x780")
        self.minsize(980, 650)
        self.df_f = self.df_h = None
        self.result_f = self.result_h = self.metrics = None
        self.models = {}
        self._build()

    def _build(self):
        header = ttk.Frame(self, padding=(16, 12))
        header.pack(fill="x")
        ttk.Label(header, text="🍍 TROPI SNACKS S.A.C.", font=("Segoe UI", 19, "bold")).pack(anchor="w")
        ttk.Label(header, text="Estimación de humedad y pérdida de masa de piña durante la deshidratación",
                  font=("Segoe UI", 10)).pack(anchor="w", pady=(3, 0))

        files = ttk.LabelFrame(self, text="1. Archivos de datos experimentales (CSV)", padding=12)
        files.pack(fill="x", padx=16, pady=5)
        self.f_path = tk.StringVar()
        self.h_path = tk.StringVar()
        row1 = ttk.Frame(files); row1.pack(fill="x", pady=3)
        ttk.Label(row1, text="Freidora de aire:", width=18).pack(side="left")
        ttk.Entry(row1, textvariable=self.f_path).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row1, text="Seleccionar…", command=lambda: self.choose_file("f")).pack(side="left")
        row2 = ttk.Frame(files); row2.pack(fill="x", pady=3)
        ttk.Label(row2, text="Horno industrial:", width=18).pack(side="left")
        ttk.Entry(row2, textvariable=self.h_path).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row2, text="Seleccionar…", command=lambda: self.choose_file("h")).pack(side="left")

        params = ttk.LabelFrame(self, text="2. Tiempos que deseas estimar (minutos separados por comas)", padding=12)
        params.pack(fill="x", padx=16, pady=5)
        self.f_times = tk.StringVar(value="12, 20, 28")
        self.h_times = tk.StringVar(value="80, 100, 120")
        r = ttk.Frame(params); r.pack(fill="x")
        ttk.Label(r, text="Freidora:", width=18).pack(side="left")
        ttk.Entry(r, textvariable=self.f_times, width=35).pack(side="left", padx=(0, 16))
        ttk.Label(r, text="Horno:").pack(side="left")
        ttk.Entry(r, textvariable=self.h_times, width=35).pack(side="left", padx=8)
        actions = ttk.Frame(self, padding=(16, 8)); actions.pack(fill="x")
        self.run_btn = ttk.Button(actions, text="Calcular estimaciones", command=self.run_calculation)
        self.run_btn.pack(side="left")
        self.export_btn = ttk.Button(actions, text="Exportar resultados a Excel", command=self.export_excel, state="disabled")
        self.export_btn.pack(side="left", padx=8)
        self.status = tk.StringVar(value="Carga los dos CSV para comenzar.")
        ttk.Label(actions, textvariable=self.status).pack(side="left", padx=10)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=16, pady=(4, 16))
        self.tab_f = ttk.Frame(self.tabs); self.tabs.add(self.tab_f, text="Freidora")
        self.tab_h = ttk.Frame(self.tabs); self.tabs.add(self.tab_h, text="Horno")
        self.tab_m = ttk.Frame(self.tabs); self.tabs.add(self.tab_m, text="Métricas")
        self.tab_plot = ttk.Frame(self.tabs); self.tabs.add(self.tab_plot, text="Gráficos 3D")
        self.tree_f = self.make_tree(self.tab_f)
        self.tree_h = self.make_tree(self.tab_h)
        self.tree_m = self.make_tree(self.tab_m)
        self.plot_frame = self.tab_plot

    def make_tree(self, parent):
        frame = ttk.Frame(parent, padding=8); frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, show="headings")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1); frame.columnconfigure(0, weight=1)
        return tree

    def choose_file(self, which):
        path = filedialog.askopenfilename(title="Selecciona el archivo CSV", filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")])
        if path:
            (self.f_path if which == "f" else self.h_path).set(path)

    def fill_tree(self, tree, df):
        tree.delete(*tree.get_children())
        cols = list(df.columns)
        tree["columns"] = cols
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(115, min(220, len(str(c))*9)), anchor="center")
        for _, row in df.iterrows():
            vals = [("" if pd.isna(v) else f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)) for v in row]
            tree.insert("", "end", values=vals)

    def run_calculation(self):
        if not self.f_path.get() or not self.h_path.get():
            messagebox.showwarning("Faltan archivos", "Selecciona el CSV de la freidora y el CSV del horno.")
            return
        try:
            self.status.set("Procesando datos y entrenando ANN…")
            self.update_idletasks()
            df_f = clean_data(read_csv_flexible(self.f_path.get()))
            df_h = clean_data(read_csv_flexible(self.h_path.get()))
            tf, th = parse_times(self.f_times.get()), parse_times(self.h_times.get())
            rf, mf, modf = calculate(df_f, tf, "Freidora de aire")
            rh, mh, modh = calculate(df_h, th, "Horno industrial")
            self.df_f, self.df_h = df_f, df_h
            self.result_f, self.result_h = rf, rh
            self.metrics = pd.concat([mf, mh], ignore_index=True)
            self.models = {"Freidora de aire": modf, "Horno industrial": modh}
            self.fill_tree(self.tree_f, rf.drop(columns=["Equipo"]))
            self.fill_tree(self.tree_h, rh.drop(columns=["Equipo"]))
            self.fill_tree(self.tree_m, self.metrics)
            self.draw_plots()
            self.export_btn.configure(state="normal")
            self.status.set("Cálculo completado.")
            messagebox.showinfo("Listo", "Estimaciones calculadas correctamente.")
        except Exception as e:
            self.status.set("No se pudo completar el cálculo.")
            messagebox.showerror("Error al procesar", str(e))

    def draw_plots(self):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        fig = Figure(figsize=(10, 6), dpi=90)
        for idx, equipment in enumerate(("Freidora de aire", "Horno industrial"), start=1):
            m = self.models[equipment]
            ax = fig.add_subplot(1, 2, idx, projection="3d")
            x, h, p = m["x"], m["h"], m["p"]
            xx = np.linspace(float(x.min()), float(x.max()), 180)
            ax.scatter(x, p, h, color="black", label="Datos reales", s=20)
            ax.plot(xx, m["lp"](xx), m["lh"](xx), label="Lagrange")
            ax.plot(xx, m["sp"](xx), m["sh"](xx), label="Spline cúbico")
            ax.plot(xx, ann_predict(x, p, xx), ann_predict(x, h, xx), linestyle="--", label="ANN")
            ax.set_title(equipment)
            ax.set_xlabel("Tiempo (min)")
            ax.set_ylabel("Pérdida de masa (%)")
            ax.set_zlabel("Humedad (%)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def export_excel(self):
        if self.result_f is None:
            return
        path = filedialog.asksaveasfilename(title="Guardar resultados", defaultextension=".xlsx",
                                            initialfile="Resultados_Tropi_Snacks.xlsx",
                                            filetypes=[("Libro de Excel", "*.xlsx")])
        if not path:
            return
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                self.result_f.to_excel(writer, sheet_name="Estimaciones_Freidora", index=False)
                self.result_h.to_excel(writer, sheet_name="Estimaciones_Horno", index=False)
                self.metrics.to_excel(writer, sheet_name="Metricas", index=False)
                self.df_f.to_excel(writer, sheet_name="Datos_Freidora", index=False)
                self.df_h.to_excel(writer, sheet_name="Datos_Horno", index=False)
            messagebox.showinfo("Excel guardado", f"Resultados guardados en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error al exportar", str(e))

if __name__ == "__main__":
    TropiApp().mainloop()
