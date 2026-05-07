import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import threading
import datetime
import time
import csv 



class EnergyAI:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_model = None
        self.iso_forest = None
        self.look_back = 24
        self.is_trained = False
        
        
        self.real_data = None
        self.validation_data = None 
        self.current_play_index = 0
        self.use_real_data = False
        self.dataset_name = "Synthetic Generator"
        
        
        self.training_history = {'loss': []}
        self.metrics = {'rmse': 0.0, 'mae': 0.0}

    def load_uci_dataset(self, filepath, log_callback):
        """Loads the official UCI Household Power Consumption Dataset."""
        try:
            log_callback("Loading Dataset (This may take a moment)...")
            
            
            df = pd.read_csv(filepath, sep=';', nrows=30000, 
                             low_memory=False, na_values=['?', 'nan'])
            
            log_callback("Preprocessing Data (Cleaning & Normalization)...")
            df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
            df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
            df.dropna(subset=['Global_active_power'], inplace=True)
            df.set_index('dt', inplace=True)
            
            
            df_hourly = df['Global_active_power'].resample('h').mean().dropna()
            
            self.real_data = pd.DataFrame(df_hourly)
            self.use_real_data = True
            self.dataset_name = "UCI Household Power (Real-World)"
            return True
            
        except Exception as e:
            log_callback(f"Error loading data: {e}")
            print(f"Debug Error: {e}")
            return False

    def train_models(self, log_callback):
        """Trains models and captures performance metrics."""
        if not self.use_real_data:
            log_callback("No File. Generating Synthetic Data for Demo...")
            df = self.generate_synthetic_data()
        else:
            df = self.real_data

        data = df.values
        
        
        train_size = int(len(data) * 0.60)
        train_data = data[:train_size]
        test_data = data[train_size:]
        self.validation_data = test_data 
        
        
        log_callback("Training Anomaly Detector (Isolation Forest)...")
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.iso_forest.fit(train_data)
        
        
        log_callback("Training Forecasting Model (LSTM - Edge Optimized)...")
        scaled_train = self.scaler.fit_transform(train_data)
        
        X_train, y_train = [], []
        for i in range(self.look_back, len(scaled_train)):
            X_train.append(scaled_train[i-self.look_back:i, 0])
            y_train.append(scaled_train[i, 0])
            
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
        self.lstm_model.add(Dense(1))
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
        
        history = self.lstm_model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
        self.training_history['loss'] = history.history['loss']
        
        
        log_callback("Calculating Edge Model Accuracy...")
        scaled_test = self.scaler.transform(test_data[:500])
        X_test, y_test = [], []
        for i in range(self.look_back, len(scaled_test)):
            X_test.append(scaled_test[i-self.look_back:i, 0])
            y_test.append(scaled_test[i, 0])
        
        if len(X_test) > 0:
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            y_test = np.array(y_test)
            
            preds = self.lstm_model.predict(X_test, verbose=0)
            inv_preds = self.scaler.inverse_transform(preds)
            inv_y = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            self.metrics['rmse'] = np.sqrt(mean_squared_error(inv_y, inv_preds))
            self.metrics['mae'] = mean_absolute_error(inv_y, inv_preds)
            log_callback(f"Model Accuracy -> RMSE: {self.metrics['rmse']:.4f}")

        self.is_trained = True
        log_callback("System Ready. Edge AI Models Active.")

    def generate_synthetic_data(self, days=60):
        date_range = pd.date_range(start='2024-01-01', periods=days*24, freq='h')
        time_step = np.arange(len(date_range))
        daily = 2 * np.sin(2 * np.pi * time_step / 24)
        noise = np.random.normal(0, 0.3, len(date_range))
        power = np.maximum(3.0 + daily + noise, 0.2)
        return pd.DataFrame({'Global_active_power': power}, index=date_range)

    def get_next_reading(self):
        if not self.use_real_data:
            return max(0.2, 3.0 + np.random.normal(0, 0.5))
        if self.current_play_index >= len(self.validation_data):
            self.current_play_index = 0 
        val = self.validation_data[self.current_play_index][0]
        self.current_play_index += 1
        return val

    def predict_next(self, recent_data):
        if not self.is_trained or len(recent_data) < self.look_back: return 0.0, 0.0
        input_seq = np.array(recent_data[-self.look_back:]).reshape(-1, 1)
        scaled_seq = self.scaler.transform(input_seq)
        X_input = scaled_seq.reshape(1, self.look_back, 1)
        
        start_time = time.time()
        prediction_scaled = self.lstm_model.predict(X_input, verbose=0)
        latency_ms = (time.time() - start_time) * 1000
        
        return self.scaler.inverse_transform(prediction_scaled)[0][0], latency_ms

    def check_anomaly(self, current_reading):
        if not self.is_trained: return 1
        return self.iso_forest.predict([[current_reading]])[0]



class SmartMeterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Powered Smart Energy Meter (Edge & Cloud Integrated)")
        self.root.geometry("1350x950")
        self.root.configure(bg="#1e1e1e") 
        
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2d3436", foreground="white", padding=[15, 5], font=('Segoe UI', 10))
        style.map("TNotebook.Tab", background=[("selected", "#00cec9")], foreground=[("selected", "black")])
        style.configure("TFrame", background="#1e1e1e")

        
        style.configure("TButton.EcoOn.TButton", background="#00b894", foreground="black", font=('Segoe UI', 10, 'bold'))
        style.map("TButton.EcoOn.TButton", background=[("active", "#00cec9")])
        
        self.ai = EnergyAI()
        self.running = False
        self.eco_mode = False
        self.current_time = datetime.datetime.now()
        
        
        self.history_data = [] 
        self.total_kwh = 1420.5 
        self.current_bill = 450.0
        
        
        self.peak_kwh = 0.0
        self.off_peak_kwh = 0.0
        self.co2_emissions = 0.0 
        
        
        self.x_dates = []
        self.y_actual = []
        self.y_pred = []
        self.y_cost = []
        self.appliance_stats = {"AC/Heater": 0, "Kitchen": 0, "Lights/TV": 0, "Always On": 0}
        
        
        self.temp_spike = 0.0

        self.create_layout()
        threading.Thread(target=self.initialize_system, daemon=True).start()

    def create_layout(self):
        
        header = tk.Frame(self.root, bg="#0984e3", height=80)
        header.pack(fill="x")
        
        title_frame = tk.Frame(header, bg="#0984e3")
        title_frame.pack(side="left", padx=20)
        tk.Label(title_frame, text="AI SMART METER", bg="#0984e3", fg="white", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        tk.Label(title_frame, text="Forecasting | Anomaly Detection | Consumer Feedback", bg="#0984e3", fg="#dfe6e9", font=("Segoe UI", 10)).pack(anchor="w")

        info_frame = tk.Frame(header, bg="#0984e3")
        info_frame.pack(side="right", padx=20)
        self.lbl_clock = tk.Label(info_frame, text="00:00:00", bg="#0984e3", fg="white", font=("Courier New", 18, "bold"))
        self.lbl_clock.pack(anchor="e")
        self.lbl_source = tk.Label(info_frame, text="Source: Initializing...", bg="#0984e3", fg="#81ecec", font=("Arial", 10, "bold"))
        self.lbl_source.pack(anchor="e")

        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        
        self.tab_dashboard = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dashboard, text="  Digital Twin Dashboard  ")
        self.build_dashboard_tab()

        
        self.tab_analytics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analytics, text="  Analytics & Edge Performance  ")
        self.build_analytics_tab()

        
        self.tab_settings = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_settings, text="  Cloud & System Config  ")
        self.build_settings_tab()

        
        log_frame = tk.Frame(self.root, bg="#2d3436", height=100)
        log_frame.pack(fill="x", side="bottom")
        tk.Label(log_frame, text="SYSTEM LOG:", bg="#2d3436", fg="#00cec9", font=("Consolas", 9, "bold")).pack(anchor="w", padx=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=5, bg="#2d3436", fg="#dfe6e9", font=("Consolas", 9), state='disabled')
        self.log_text.pack(fill="both", expand=True, padx=5, pady=2)

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {msg}\n"
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, full_msg)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    
    def build_dashboard_tab(self):
        metrics_frame = tk.Frame(self.tab_dashboard, bg="#2d3436", bd=2, relief="groove")
        metrics_frame.pack(fill="x", padx=10, pady=10)
        
        self.lbl_power = self.add_metric_card(metrics_frame, "CURRENT LOAD", 0, "kW", "#ff7675")
        self.lbl_energy = self.add_metric_card(metrics_frame, "TOTAL ENERGY", 1, "kWh", "#55efc4")
        self.lbl_bill = self.add_metric_card(metrics_frame, "CURRENT BILL", 2, "₹", "#ffeaa7")
        
        feedback_frame = tk.Frame(metrics_frame, bg="#2d3436", width=300)
        feedback_frame.grid(row=0, column=3, padx=10, sticky="nsew")
        tk.Label(feedback_frame, text="AI RECOMMENDATIONS (Consumer Feedback Loop)", bg="#2d3436", fg="#74b9ff", font=("Arial", 9, "bold")).pack(anchor="nw")
        self.lbl_recommendation = tk.Label(feedback_frame, text="System initializing. Awaiting first data stream...", bg="#2d3436", fg="white", font=("Arial", 10, "italic"), wraplength=250, justify="left")
        self.lbl_recommendation.pack(fill="both", expand=True)
        metrics_frame.grid_columnconfigure(3, weight=1)

        
        plot_frame = tk.Frame(self.tab_dashboard, bg="#1e1e1e")
        plot_frame.pack(fill="both", expand=True, padx=10)
        
        self.fig_dash, (self.ax_live, self.ax_cost) = plt.subplots(1, 2, figsize=(10, 4), facecolor='#1e1e1e')
        
        self.ax_live.set_facecolor('#2d3436')
        self.ax_live.set_title('Real-Time Digital Twin (Actual vs. Forecast)', color='white', fontsize=10)
        self.ax_live.tick_params(colors='white')
        self.line_actual, = self.ax_live.plot([], [], 'g-', linewidth=1.5, label='Actual')
        self.line_pred, = self.ax_live.plot([], [], 'm--', linewidth=1.5, label='Forecast')
        self.ax_live.legend(facecolor='#2d3436', labelcolor='white', loc='upper left')
        self.ax_live.grid(True, alpha=0.1)

        self.ax_cost.set_facecolor('#2d3436')
        self.ax_cost.set_title('Cumulative Cost Analysis', color='white', fontsize=10)
        self.ax_cost.tick_params(colors='white')
        self.line_cost, = self.ax_cost.plot([], [], 'y-', linewidth=1.5)
        self.ax_cost.grid(True, alpha=0.1)
        
        self.canvas_dash = FigureCanvasTkAgg(self.fig_dash, master=plot_frame)
        self.canvas_dash.get_tk_widget().pack(fill="both", expand=True)

        
        ctrl_frame = tk.Frame(self.tab_dashboard, bg="#1e1e1e")
        ctrl_frame.pack(fill="x", pady=10)
        
        self.btn_start = ttk.Button(ctrl_frame, text="Start Simulation", command=self.toggle_simulation)
        self.btn_start.pack(side="left", padx=20)
        
        
        self.btn_eco = ttk.Button(ctrl_frame, text="🌿 Enable Eco-Mode", command=self.toggle_eco_mode)
        self.btn_eco.pack(side="left", padx=10)
        
        ttk.Button(ctrl_frame, text="⚠️ Simulate Theft", command=self.inject_theft).pack(side="left", padx=10)
        ttk.Button(ctrl_frame, text="📄 Generate Bill", command=self.generate_bill_popup).pack(side="right", padx=20)
        
        self.lbl_latency = tk.Label(ctrl_frame, text="Edge Latency: -- ms", bg="#1e1e1e", fg="#b2bec3", font=("Consolas", 9))
        self.lbl_latency.pack(side="right", padx=10)

    def add_metric_card(self, parent, title, col, unit, color):
        frame = tk.Frame(parent, bg="#2d3436")
        frame.grid(row=0, column=col, padx=10, pady=15, sticky="nsew")
        tk.Label(frame, text=title, bg="#2d3436", fg="#dfe6e9", font=("Arial", 8)).pack()
        val = tk.Label(frame, text="0.00", bg="#2d3436", fg=color, font=("Segoe UI", 20, "bold"))
        val.pack()
        tk.Label(frame, text=unit, bg="#2d3436", fg="#dfe6e9", font=("Arial", 8)).pack()
        parent.grid_columnconfigure(col, weight=1)
        return val

    
    def build_analytics_tab(self):
        stats_frame = tk.Frame(self.tab_analytics, bg="#1e1e1e")
        stats_frame.pack(fill="x", padx=20, pady=20)
        
        self.lbl_rmse = tk.Label(stats_frame, text="RMSE: N/A", bg="#1e1e1e", fg="#fdcb6e", font=("Segoe UI", 12, "bold"))
        self.lbl_rmse.pack(side="left", padx=20)
        
        self.lbl_co2 = tk.Label(stats_frame, text="CO2 Footprint: 0.0 kg", bg="#1e1e1e", fg="#00b894", font=("Segoe UI", 12, "bold"))
        self.lbl_co2.pack(side="right", padx=20)
        
        plot_frame = tk.Frame(self.tab_analytics, bg="#1e1e1e")
        plot_frame.pack(fill="both", expand=True)
        
        self.fig_anl, (self.ax_loss, self.ax_pie) = plt.subplots(1, 2, figsize=(10, 4), facecolor='#1e1e1e')
        
        self.ax_loss.set_facecolor('#2d3436')
        self.ax_loss.set_title('Model Training Loss', color='white')
        self.ax_loss.tick_params(colors='white')
        self.ax_loss.grid(True, alpha=0.1)

        self.ax_pie.set_facecolor('#1e1e1e')
        self.ax_pie.set_title('Disaggregated Load Profile', color='white')
        
        self.canvas_anl = FigureCanvasTkAgg(self.fig_anl, master=plot_frame)
        self.canvas_anl.get_tk_widget().pack(fill="both", expand=True)

    def update_analytics_plots(self):
        loss_data = self.ai.training_history.get('loss', [])
        if loss_data:
            self.ax_loss.clear()
            self.ax_loss.set_facecolor('#2d3436')
            self.ax_loss.set_title('Model Training Loss', color='white')
            self.ax_loss.tick_params(colors='white')
            self.ax_loss.plot(loss_data, 'c-o', linewidth=2)
            self.ax_loss.grid(True, alpha=0.1)
            self.lbl_rmse.config(text=f"Model RMSE: {self.ai.metrics['rmse']:.4f}")

        
        self.lbl_co2.config(text=f"CO2 Footprint: {self.co2_emissions:.2f} kg")

        labels = list(self.appliance_stats.keys())
        sizes = list(self.appliance_stats.values())
        if sum(sizes) == 0: sizes = [1, 1, 1, 1] 
        
        self.ax_pie.clear()
        self.ax_pie.set_title('Disaggregated Load Profile', color='white')
        self.ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', 
                         colors=['#ff7675', '#74b9ff', '#55efc4', '#ffeaa7'],
                         textprops={'color':"white"})
        self.canvas_anl.draw()

    
    def build_settings_tab(self):
        f = tk.Frame(self.tab_settings, bg="#1e1e1e")
        f.pack(fill="both", expand=True, padx=50, pady=50)
        
        tk.Label(f, text="Simulation Speed (1 real-sec = X virtual-mins):", bg="#1e1e1e", fg="white", font=("Arial", 12, "bold")).pack(anchor="w")
        self.slider_speed = ttk.Scale(f, from_=1, to=60, orient="horizontal", length=300)
        self.slider_speed.set(30)
        self.slider_speed.pack(anchor="w", pady=5)
        
        ttk.Button(f, text="Export Data (CSV)", command=self.export_data).pack(anchor="w", pady=20)

    def export_data(self):
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV/TXT", "*.csv;*.txt")])
        if filename:
            try:
                
                if not self.x_dates:
                    messagebox.showerror("Error", "No data recorded to export.")
                    return
                
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Load (kW)", "Cost (INR)"])
                    for i in range(len(self.x_dates)):
                        writer.writerow([self.x_dates[i].strftime('%Y-%m-%d %H:%M:%S'), self.y_actual[i], self.y_cost[i]])
                messagebox.showinfo("Success", f"Data Exported Successfully to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")

    
    def initialize_system(self):
        self.btn_start.config(state="disabled")
        self.log("System Booting...")
        
        is_real = messagebox.askyesno("Dataset Selection", 
                                     "Load Real UCI Household Dataset?\n\nYES: Use Actual Data (Recommended)\nNO: Use Synthetic Data (Fast Demo)")
        if is_real:
            path = filedialog.askopenfilename(title="Select UCI Dataset", filetypes=[("CSV/TXT", "*.csv;*.txt")])
            if path:
                if self.ai.load_uci_dataset(path, self.log):
                    self.log("UCI Dataset Loaded Successfully.")
                else:
                    self.log("Failed to load. Reverting to Synthetic.")
            else:
                self.log("No file selected. Reverting to Synthetic.")
        
        self.lbl_source.config(text=f"Source: {self.ai.dataset_name}")
        self.ai.train_models(self.log)
        
        self.history_data = [self.ai.get_next_reading() for _ in range(self.ai.look_back)] 
        self.update_analytics_plots() 
        self.root.after(0, lambda: self.btn_start.config(state="normal"))
        self.root.after(0, lambda: messagebox.showinfo("Ready", "System Online. Click 'Start Simulation'."))

    def toggle_simulation(self):
        if self.running:
            self.running = False
            self.btn_start.config(text="Resume Simulation")
            self.log("Simulation Paused.")
        else:
            self.running = True
            self.btn_start.config(text="Pause Simulation")
            self.log("Simulation Started.")
            self.run_tick()

    def toggle_eco_mode(self):
        self.eco_mode = not self.eco_mode
        if self.eco_mode:
            
            self.btn_eco.config(text="🌿 Eco-Mode: ON (Click to Disable)", style="TButton.EcoOn.TButton") 
            self.log("Eco-Mode ENABLED: AI will optimize peak loads for Demand Response.")
        else:
            
            self.btn_eco.config(text="🌿 Enable Eco-Mode", style="TButton")
            self.log("Eco-Mode DISABLED.")

    def inject_theft(self):
        self.temp_spike = 8.0
        
        self.root.after(1500, lambda: setattr(self, 'temp_spike', 0)) 
        self.log("WARNING: Manual Anomaly Injection Triggered. Load spike initiated.")
        messagebox.showwarning("Security Alert", "Theft Pattern Detected! Observing sudden, large spike in load.")

    def generate_bill_popup(self):
        
        bill_window = tk.Toplevel(self.root)
        bill_window.title("Monthly Electricity Bill")
        bill_window.geometry("400x500")
        bill_window.configure(bg="white")
        
        tk.Label(bill_window, text="ELECTRICITY BILL RECEIPT", font=("Arial", 16, "bold"), bg="white").pack(pady=10)
        tk.Label(bill_window, text=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", bg="white").pack()
        tk.Label(bill_window, text="Account ID: AI-METER-007", bg="white").pack()
        tk.Label(bill_window, text="-----------------------------------------", bg="white").pack()
        
        content_frame = tk.Frame(bill_window, bg="white")
        content_frame.pack(fill="x", padx=20, pady=10)
        
        def add_row(label, val, bold=False):
            f = tk.Frame(content_frame, bg="white")
            f.pack(fill="x", pady=2)
            font_style = ("Arial", 10, "bold") if bold else ("Arial", 10)
            tk.Label(f, text=label, bg="white", font=font_style).pack(side="left")
            tk.Label(f, text=val, bg="white", font=font_style).pack(side="right")
            
        add_row("Total Energy Consumed:", f"{self.total_kwh:.2f} kWh")
        add_row("  - Peak Hours (17:00-23:00) @ 9.0 INR/kWh:", f"{self.peak_kwh:.2f} kWh")
        add_row("  - Off-Peak Hours @ 5.0 INR/kWh:", f"{self.off_peak_kwh:.2f} kWh")
        
        
        peak_cost = self.peak_kwh * 9.0
        off_peak_cost = self.off_peak_kwh * 5.0
        
        add_row("Cost - Peak:", f"₹ {peak_cost:.2f}")
        add_row("Cost - Off-Peak:", f"₹ {off_peak_cost:.2f}")
        
        tk.Label(bill_window, text="-----------------------------------------", bg="white").pack()
        
        add_row("TOTAL AMOUNT PAYABLE:", f"₹ {self.current_bill:.2f}", bold=True)
        
        tk.Label(bill_window, text="-----------------------------------------", bg="white").pack()
        add_row("Environmental Metric:", f" ")
        add_row("Carbon Footprint Generated:", f"{self.co2_emissions:.2f} kg CO2")
        
        tk.Button(bill_window, text="Acknowledge", command=bill_window.destroy).pack(pady=20)

    def update_appliance_stats(self, power):
        
        cat = "Always On"
        if power > 4.5: cat = "AC/Heater"
        elif power > 2.0: cat = "Kitchen"
        elif power > 0.5: cat = "Lights/TV"
        
        
        if power > 0.1:
            self.appliance_stats[cat] += 1
            
        if len(self.history_data) % 20 == 0:
            self.update_analytics_plots()

    def update_recommendation(self, current_load, anomaly, is_peak, next_prediction):
        """Provides actionable consumer feedback (SAFE, WATCH, URGENT status)."""
        
        
        if anomaly == -1:
            color = "red"
            recommendation = "🔴 **URGENT: ANOMALY/THEFT DETECTED!** Check high-load appliances immediately. Security alert flagged to utility."
        
        
        elif is_peak and current_load > 5.5:
            color = "#fdcb6e"
            recommendation = "🟡 **WATCH: HIGH PEAK LOAD.** Usage is very high during expensive peak hours. **ACTION:** Delay high-draw appliances (AC, EV charging) until after 11 PM."
        
        elif is_peak and next_prediction > 5.0 and self.eco_mode:
             color = "#00b894"
             recommendation = f"🌿 **ECO-MODE OPTIMIZING.** AI predicts high load ({next_prediction:.1f}kW). Demand Response is active, load is being managed to save cost."

        elif not is_peak and current_load > 4.5:
            color = "#fdcb6e"
            recommendation = "🟡 **WATCH: High Off-Peak Load.** Unusually high consumption (e.g., Water Heater, always-on server). **REVIEW:** Check for phantom loads or unexpected device schedules."
        
        
        else:
            color = "#55efc4"
            
            if current_load < 0.3:
                 recommendation = "🟢 **SAFE: Low Standby Load.** Energy consumption is minimal. Good job reducing 'phantom' loads."
            elif is_peak and current_load < 4.0:
                 recommendation = f"🟢 **SAFE: Managing Peak.** Your current peak usage is moderate. Predicted load is {next_prediction:.1f}kW. Continue monitoring."
            else:
                 recommendation = "🟢 **SAFE: Optimal Operation.** Consumption patterns are normal and follow forecast. Analyzing historical data for deep insights..."
        
        
        self.lbl_recommendation.config(text=recommendation, fg=color)

    def run_tick(self):
        if not self.running: return

        mins = int(self.slider_speed.get())
        self.current_time += datetime.timedelta(minutes=mins)
        self.lbl_clock.config(text=self.current_time.strftime("%Y-%m-%d %H:%M"))

        current_load = self.ai.get_next_reading()
        
        if self.temp_spike > 0: 
            current_load += self.temp_spike

        
        is_peak = 17 <= self.current_time.hour <= 23

        
        self.history_data.append(current_load)
        if len(self.history_data) > self.ai.look_back: self.history_data.pop(0)
        
        prediction, latency = self.ai.predict_next(self.history_data)
        
        
        if self.eco_mode and is_peak and prediction > 5.0: 
             savings = current_load * 0.20 
             current_load -= savings
             if int(self.current_time.minute) == 0: 
                  self.log(f"Eco-Mode Active: Reduced load by {savings:.2f}kW (Optimized AC/Heating)")

        
        anomaly = self.ai.check_anomaly(current_load)

        
        kwh_inc = current_load * (mins/60.0)
        rate = 9.0 if is_peak else 5.0 
        
        self.total_kwh += kwh_inc
        if is_peak: self.peak_kwh += kwh_inc
        else: self.off_peak_kwh += kwh_inc
            
        self.current_bill += kwh_inc * rate
        self.co2_emissions += kwh_inc * 0.85 
        
        self.update_appliance_stats(current_load)
        self.update_recommendation(current_load, anomaly, is_peak, prediction) 

        
        self.lbl_power.config(text=f"{current_load:.2f}")
        self.lbl_energy.config(text=f"{self.total_kwh:.1f}")
        self.lbl_bill.config(text=f"{self.current_bill:.2f}")
        self.lbl_latency.config(text=f"Inference Time: {latency:.1f} ms")
        
        if anomaly == -1:
            self.log(f"ALERT: ANOMALY detected at {self.current_time.strftime('%H:%M')} (Load: {current_load:.2f}kW)")
            self.lbl_power.config(fg="red")
        else:
            self.lbl_power.config(fg="#ff7675") 

        
        self.x_dates.append(self.current_time)
        self.y_actual.append(current_load)
        self.y_pred.append(prediction)
        self.y_cost.append(self.current_bill)
        
        if len(self.x_dates) > 50:
            self.x_dates.pop(0); self.y_actual.pop(0); self.y_pred.pop(0); self.y_cost.pop(0)

        self.line_actual.set_data(self.x_dates, self.y_actual)
        self.line_pred.set_data(self.x_dates, self.y_pred)
        self.line_cost.set_data(self.x_dates, self.y_cost)
        
        self.ax_live.relim(); self.ax_live.autoscale_view()
        self.ax_cost.relim(); self.ax_cost.autoscale_view()
        self.ax_live.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax_cost.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.canvas_dash.draw()

        self.root.after(500, self.run_tick)

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartMeterApp(root)
    root.mainloop()