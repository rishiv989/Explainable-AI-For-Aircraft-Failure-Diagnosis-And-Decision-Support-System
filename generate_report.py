from fpdf import FPDF
import datetime

class PDF(FPDF):
    def header(self):
        # Logo placeholder (optional)
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Cortex XAI Engine Monitoring System', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'Project Technical Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - Generated on {datetime.date.today()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()

# 1. Executive Summary
pdf.chapter_title('1. Executive Summary')
pdf.chapter_body(
    "The Cortex XAI Engine Monitoring System is a state-of-the-art predictive maintenance solution designed "
    "for aircraft turbofan engines. By leveraging the NASA CMAPSS dataset and advanced Machine Learning "
    "(XGBoost, Random Forest), the system accurately predicts the Remaining Useful Life (RUL) and Flight "
    "Worthiness of engines. Crucially, it incorporates eXplainable AI (XAI) techniques (LIME, SHAP) to "
    "provide transparent, actionable insights for maintenance technicians, solving the 'black box' problem "
    "in AI adoption. The system features a decoupling architecture with a fast Python backend and a modern "
    "React 'AeroGlass' frontend."
)

# 2. System Architecture
pdf.chapter_title('2. System Architecture')
pdf.chapter_body(
    "The system follows a modern Client-Server architecture:\n\n"
    "A. Backend (FastAPI):\n"
    "   - Serves as the computational core.\n"
    "   - Handles data ingestion (CMAPSS FD001-FD004).\n"
    "   - Executes ML inference pipeline.\n"
    "   - performs real-time 'What-If' simulations.\n"
    "   - Exposes RESTful API endpoints (e.g., /predict_explain, /predict_simulated).\n\n"
    "B. Frontend (React + Vite):\n"
    "   - Provides an interactive 'AeroGlass' dashboard.\n"
    "   - Visualizes live telemetry with Recharts.\n"
    "   - Features a Fleet Health Monitoring Sidebar.\n"
    "   - Includes a Simulation Panel for variable parameter adjustment."
)

# 3. Technical Implementation
pdf.chapter_title('3. Technical Implementation')
pdf.chapter_body(
    "Data Pipeline:\n"
    "Raw sensor data (Temperature, Pressure, Fan Speeds) is preprocessed to remove noise and filtered based "
    "on operating conditions. Constant sensors (e.g., Setting 3) are dropped to improve model stability.\n\n"
    "Machine Learning Models:\n"
    "1. RUL Regressor: An XGBoost model trained to predict the precise number of remaining flight cycles.\n"
    "2. State Classifier: A Random Forest classifier that categorizes engine health into 'Normal', 'Warning', "
    "or 'Critical' states.\n\n"
    "Explainable AI (XAI):\n"
    "The system uses LIME (Local Interpretable Model-agnostic Explanations) to perturb input features and "
    "identify which specific sensors contributed most to a prediction. This allows the efficient generation "
    "of text-based Maintenance Recommendations (e.g., 'Inspect HPT Coolant Bleed due to high variance')."
)

# 4. Key Features
pdf.chapter_title('4. Key Features')
pdf.chapter_body(
    "- Predictive Maintenance: Forewarns of failures cycle-by-cycle.\n"
    "- Diagnostics Forensics: Pinpoints root causes (e.g., 'Sensor 11 High').\n"
    "- What-If Simulation: Allows engineers to simulate stressors (e.g., +5% Heat) and see immediate RUL impact.\n"
    "- Fleet Monitoring: Single-pane-of-glass view for managing multiple assets.\n"
    "- AeroGlass UI: Dark-themed, high-contrast aesthetics optimized for control room environments."
)

# 5. Conclusion
pdf.chapter_title('5. Conclusion')
pdf.chapter_body(
    "The Cortex XAI System successfully demonstrates how advanced AI can be bridged with human-centric design. "
    "By moving beyond simple predictions to providing 'Simulations' and 'Explanations', it empowers operators "
    "to make data-driven safety decisions with high confidence."
)

pdf.output('Cortex_XAI_Project_Report.pdf', 'F')
print("PDF generated successfully.")
