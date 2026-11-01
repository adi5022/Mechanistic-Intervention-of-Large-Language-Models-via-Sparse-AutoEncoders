"""
Generate the complete, publication-quality academic submission PDF:
A2_B23CS2114_AlvinBinu.pdf

Follows the exact question-wise structure, typography, table styling, and visual
fidelity of the BioMarkHER2 sample submission.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_FILENAME = "A2_B23CS2114_AlvinBinu.pdf"

class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically and adds running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress running header/footer on cover page

        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#555555"))

        # Running Header
        self.drawRightString(A4[0] - 40, A4[1] - 30, "Software Engineering Assignment 2")
        self.drawString(40, A4[1] - 30, "FeatureScalpel (B23CS2114 — Alvin Binu)")
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.6)
        self.line(40, A4[1] - 35, A4[0] - 40, A4[1] - 35)

        # Running Footer
        self.drawString(40, 25, "Department of Computer Science & Engineering · MBCET (Autonomous)")
        self.drawRightString(A4[0] - 40, 25, f"Page {self._pageNumber} of {page_count}")
        self.line(40, 35, A4[0] - 40, 35)

        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_FILENAME,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0D3B66'),
        alignment=1,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#2E4057'),
        alignment=1,
        spaceAfter=6
    )

    college_header = ParagraphStyle(
        'CollegeHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1F2328'),
        alignment=1,
        spaceAfter=4
    )

    college_sub = ParagraphStyle(
        'CollegeSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#555555'),
        alignment=1,
        spaceAfter=15
    )

    q_title = ParagraphStyle(
        'QuestionTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0D3B66'),
        spaceBefore=12,
        spaceAfter=6
    )

    q_prompt = ParagraphStyle(
        'QuestionPrompt',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#1F2328'),
        spaceAfter=6
    )

    body = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#24292F'),
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        'BodyDarkBold',
        parent=body,
        fontName='Helvetica-Bold'
    )

    caption = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#57606A'),
        alignment=1,
        spaceBefore=4,
        spaceAfter=10
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#1F2328')
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("Mar Baselios College of Engineering and Technology", college_header))
    story.append(Paragraph("Department of Computer Science and Engineering<br/>(Autonomous Institution Affiliated to APJ Abdul Kalam Technological University)", college_sub))
    story.append(Spacer(1, 30))

    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor("#0D3B66"), spaceBefore=10, spaceAfter=20))
    story.append(Paragraph("ASSIGNMENT 2 REPORT", title_style))
    story.append(Paragraph("Main Project on GitHub: Testing, CI/CD Pipeline and Project Management", subtitle_style))
    story.append(Paragraph("Course: 23CTB40B | Software Engineering", college_sub))
    story.append(HRFlowable(width="80%", thickness=1, color=colors.HexColor("#0D3B66"), spaceBefore=10, spaceAfter=30))

    story.append(Spacer(1, 40))
    story.append(Paragraph("<b>Submitted by</b>", ParagraphStyle('SubHeading', parent=body, fontSize=11, fontName='Helvetica-Bold', textColor=colors.HexColor('#0D3B66'))))
    story.append(Spacer(1, 8))

    meta_data = [
        [Paragraph("<b>Name of the student</b>", body), Paragraph("ALVIN BINU", body_bold)],
        [Paragraph("<b>University roll number</b>", body), Paragraph("B23CS2114", body)],
        [Paragraph("<b>KTU registration number</b>", body), Paragraph("MBT23CA014", body)],
        [Paragraph("<b>Branch & Course</b>", body), Paragraph("Computer Science and Engineering (Artificial Intelligence) [CT]", body)],
        [Paragraph("<b>Main Project Title</b>", body), Paragraph("<b>FeatureScalpel</b>: Mechanistic Interpretability and Intervention of Large Language Models via Sparse Autoencoders", body)],
        [Paragraph("<b>GitHub repository URL</b>", body), Paragraph('<font color="#0969DA"><u>https://github.com/adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders</u></font>', body)],
        [Paragraph("<b>Individual GitHub account</b>", body), Paragraph("<b>Alvinbinu</b> (<font color='#0969DA'><u>https://github.com/Alvinbinu</u></font>)", body)],
        [Paragraph("<b>Assigned Work Scope</b>", body), Paragraph("Streamlit dashboard validation, PyTest test suite (14 TCs with BVA/EP), GitHub Actions CI/CD pipeline", body)],
    ]

    meta_table = Table(meta_data, colWidths=[150, 360])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
        ('BOX', (0, 0), (-1, -1), 1.2, colors.HexColor("#0D3B66")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)

    story.append(Spacer(1, 50))
    story.append(Paragraph("Thiruvananthapuram, Kerala · October / November 2026", college_sub))
    story.append(PageBreak())

    # =========================================================================
    # QUESTION 1
    # =========================================================================
    story.append(Paragraph("Question - 1", q_title))
    story.append(Paragraph("Upload your Main Project to GitHub and manage it as a team. [10 Marks]", q_prompt))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D3B66"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("<b>Answer</b>", body_bold))
    story.append(Paragraph(
        "<b>Team Collaboration and Traceability:</b> Every project milestone is mapped to a tracked GitHub Issue, "
        "every team member develops on a dedicated feature branch isolated from the research branches, and every branch "
        "merges via a reviewed and approved Pull Request. Integration work is strictly isolated on <code>se_assignment</code> "
        "to guarantee that the core research codebase and benchmark results remain intact.", body
    ))

    # Traceability Table
    trace_data = [
        [Paragraph("<b>Member</b>", body_bold), Paragraph("<b>Issues</b>", body_bold), Paragraph("<b>Branch</b>", body_bold),
         Paragraph("<b>Commits</b>", body_bold), Paragraph("<b>Pull Request</b>", body_bold), Paragraph("<b>Approved By</b>", body_bold)],
        [Paragraph("Alvin Binu", body), Paragraph("#3", body), Paragraph("feature/alvin-app-testing", code_style),
         Paragraph("2 (2c89532, 2fbe101)", body), Paragraph("#1 (merged)", body_bold), Paragraph("adi5022 (Adithyan)", body)],
        [Paragraph("Adithyan R S", body), Paragraph("#1", body), Paragraph("feature/causal-intervention", code_style),
         Paragraph("2 (pending)", body), Paragraph("#2 (pending)", body), Paragraph("Alvinbinu", body)],
        [Paragraph("Alona Biju", body), Paragraph("#2", body), Paragraph("feature/benchmark-eval", code_style),
         Paragraph("2 (pending)", body), Paragraph("#3 (pending)", body), Paragraph("DarsanM", body)],
        [Paragraph("Darsan M", body), Paragraph("#4", body), Paragraph("feature/cross-model-deploy", code_style),
         Paragraph("2 (pending)", body), Paragraph("#4 (pending)", body), Paragraph("Alonabiju", body)],
    ]
    trace_table = Table(trace_data, colWidths=[80, 45, 140, 95, 80, 75])
    trace_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(trace_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>(a) Repository Structure and .gitignore:</b> The repository is hosted publicly at "
        "<font color='#0969DA'><u>github.com/adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders</u></font>. "
        "The root <code>README.md</code> provides the formal project title, academic abstract, verified team table, complete technology stack, "
        "three-tier system architecture, Streamlit execution commands, unit test runner instructions, and the active GitHub Actions CI workflow status badge. "
        "The repository <code>.gitignore</code> strictly excludes large datasets (<code>data/counterfact_hard_set.json</code>), model checkpoints, "
        "large zip packages (<code>*.zip</code>, <code>outputs/</code>), environment files (<code>.env</code>), and virtual environments (<code>.venv/</code>). "
        "Zero API keys, credentials, or multi-gigabyte binaries are tracked in Git.", body
    ))

    story.append(Paragraph(
        "<b>(b) GitHub Issues for Pending Work:</b> Four structured issues were created for remaining engineering work and assigned to respective members: "
        "<b>#1</b> (Adithyan R S: Causal Feature Selector and Layer 8 Ablation Stability), "
        "<b>#2</b> (Alona Biju: CounterFact Hard Set Benchmark Verification and Reproducibility), "
        "<b>#3</b> (Alvin Binu: Automated Unit Test Suite and CI/CD Pipeline Integration), and "
        "<b>#4</b> (Darsan M: Cross-Model Ridge Translator and Deployment Specification).", body
    ))

    story.append(Paragraph(
        "<b>(c) Feature Branch, Commits, and Peer-Reviewed Pull Request:</b> I worked on the isolated branch <code>feature/alvin-app-testing</code> "
        "based on <code>se_assignment</code> and made two verified commits: "
        "<br/>• <b>Commit 1 (<code>2c89532</code>):</b> <i>'Add 14 unit test cases with BVA and EP for scale mapping, safety filter and probability formatting (TC01-TC14)'</i> "
        "<br/>• <b>Commit 2 (<code>2fbe101</code>):</b> <i>'Configure GitHub Actions CI pipeline and update README with status badge, team roles and run instructions'</i> "
        "<br/>Pull Request #1 was opened targeting <code>se_assignment</code>, underwent peer review, received formal approval from teammate <b>Adithyan R S (@adi5022)</b>, "
        "and was merged into <code>se_assignment</code> after automated CI tests passed.", body
    ))
    story.append(Spacer(1, 4))

    # Embed PR Review Card
    if os.path.exists("docs/assignment-2/evidence/pr_review.png"):
        story.append(Image("docs/assignment-2/evidence/pr_review.png", width=510, height=270))
        story.append(Paragraph("Figure 1: GitHub Pull Request #1 reviewed and approved by peer teammate (@adi5022) prior to merge", caption))

    story.append(PageBreak())

    # =========================================================================
    # QUESTION 2
    # =========================================================================
    story.append(Paragraph("Question - 2", q_title))
    story.append(Paragraph("Plan and track the remaining work of your Main Project. [10 Marks]", q_prompt))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D3B66"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("<b>Answer</b>", body_bold))
    story.append(Paragraph(
        "<b>(a) Kanban Board in GitHub Projects:</b> Project tasks are managed using a GitHub Projects Kanban board titled "
        "<i>'FeatureScalpel — Assignment 2 Project Management'</i> configured with standard workflow stages: <b>To Do</b>, <b>In Progress</b>, and <b>Done</b>. "
        "Issue #3 (Unit Testing & CI Pipeline) is placed in <b>Done</b> following successful PR merge. Issue #1 is <b>In Progress</b>, "
        "while Issues #2 and #4 reside in <b>To Do</b>.", body
    ))

    if os.path.exists("docs/assignment-2/evidence/kanban_board.png"):
        story.append(Image("docs/assignment-2/evidence/kanban_board.png", width=510, height=265))
        story.append(Paragraph("Figure 2: Kanban board in GitHub Projects tracking tasks across To Do, In Progress, and Done", caption))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>(b) Gantt Chart for Remaining Work:</b> An 8-task schedule spans 14 October to 12 November 2026, establishing task sequences, "
        "durations, member assignments, and milestone dependencies.", body
    ))

    if os.path.exists("docs/assignment-2/evidence/gantt_chart.png"):
        story.append(Image("docs/assignment-2/evidence/gantt_chart.png", width=510, height=240))
        story.append(Paragraph("Figure 3: Gantt chart scheduling remaining work packages, durations, and member allocations", caption))

    story.append(PageBreak())

    story.append(Paragraph(
        "<b>(c) Risk Management:</b> Three project-specific technical and operational risks were identified, analyzed, and paired with active engineering mitigations.", body
    ))

    # Risk Table
    risk_data = [
        [Paragraph("<b>ID</b>", body_bold), Paragraph("<b>Risk Description</b>", body_bold), Paragraph("<b>Type</b>", body_bold),
         Paragraph("<b>Likelihood, Impact</b>", body_bold), Paragraph("<b>Mitigation Step</b>", body_bold)],
        [
            Paragraph("<b>R1</b>", body_bold),
            Paragraph("<b>GPU VRAM Out-of-Memory (OOM):</b> Stacking 200 candidate feature ablations into a single forward pass causes CUDA allocation failure on consumer GPUs (e.g. GTX 1660 Ti 6GB).", body),
            Paragraph("Technical / Hardware", body),
            Paragraph("High, High<br/>(Score: 9)", body),
            Paragraph("Implement dynamic batch chunking via <code>MAX_EVAL_BATCH</code> in <code>src/batched_eval.py</code>. If tensor allocation exceeds safety headroom, automatically fall back to sequential CPU execution.", body)
        ],
        [
            Paragraph("<b>R2</b>", body_bold),
            Paragraph("<b>Upstream API Drift:</b> Updates to TransformerLens or SAELens alter hook registration signatures or dictionary tensor formats, breaking activation delta-patching.", body),
            Paragraph("Technical / Software", body),
            Paragraph("Medium, High<br/>(Score: 6)", body),
            Paragraph("Pin strict dependency versions in <code>requirements.txt</code> (<code>transformer-lens>=3.5.1</code>, <code>sae-lens>=6.45.3</code>) and execute multi-python CI matrix tests on every commit.", body)
        ],
        [
            Paragraph("<b>R3</b>", body_bold),
            Paragraph("<b>Cross-Backend Logit Numerical Variance:</b> Floating-point differences across CUDA, Apple MPS, and CPU cause edge-case flips in candidate safety verdicts.", body),
            Paragraph("Quality / Experimental", body),
            Paragraph("Medium, Medium<br/>(Score: 4)", body),
            Paragraph("Enforce a numerical tolerance buffer ($\Delta p \ge -10^{-6}$) in <code>src/editing.py</code> rather than strict zero, and maintain isolated deterministic PyTest suites with fixed tensor mocks.", body)
        ],
    ]

    risk_table = Table(risk_data, colWidths=[25, 140, 85, 75, 190])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 6))

    if os.path.exists("docs/assignment-2/evidence/risk_matrix.png"):
        story.append(Image("docs/assignment-2/evidence/risk_matrix.png", width=380, height=255))
        story.append(Paragraph("Figure 4: 3×3 Risk Matrix mapping risk severity (Score = Likelihood × Impact)", caption))

    story.append(PageBreak())

    # =========================================================================
    # QUESTION 3
    # =========================================================================
    story.append(Paragraph("Question - 3", q_title))
    story.append(Paragraph("Write and execute test cases for your Main Project. [15 Marks]", q_prompt))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D3B66"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("<b>Answer</b>", body_bold))
    story.append(Paragraph(
        "<b>(a) Unit Test Suite Implementation:</b> A comprehensive, lightweight unit test suite containing 14 deterministic test cases "
        "was implemented in <code>tests/test_assignment_a2.py</code>. The suite tests core mathematical functions without requiring GPU hardware, "
        "downloading 500MB+ model weights, or external network calls: "
        "<br/>1. <code>build_scale_map</code> & <code>with_extra_scale</code>: Multiplier generation for joint delta-patching hooks ($\Delta_p = \sum (s_k - 1) f_{p,k} W_{dec}[k]$). "
        "<br/>2. <code>evaluate_safety_criterion</code>: Feature safety filter logic enforcing Equation 3.4 ($safe(k) \iff r_k(t) \le r_0(t) \land p_k(t) - p_0(t) \ge -10^{-6}$). "
        "<br/>3. <code>_to_a</code>: Continuous multiplier activation bounds ($m_k = 1 + a_k \in (0, 3)$ where $a_k = -1 + 3\sigma(z_k)$, Eq. 3.5). "
        "<br/>4. <code>fmt_prob</code>: Streamlit human-readable probability and odds formatting thresholds.", body
    ))

    story.append(Paragraph("<b>(b) Test Cases and Execution Results:</b> All 14 unit test cases passed with zero failures in 2.85s.", body_bold))

    # Test Case Table (exact columns required)
    tc_data = [
        [Paragraph("<b>TC ID</b>", body_bold), Paragraph("<b>Function Tested</b>", body_bold), Paragraph("<b>Input</b>", body_bold),
         Paragraph("<b>Expected Output</b>", body_bold), Paragraph("<b>Actual Output</b>", body_bold), Paragraph("<b>Pass/Fail</b>", body_bold), Paragraph("<b>Technique</b>", body_bold)],
        [Paragraph("TC01", body), Paragraph("build_scale_map", code_style), Paragraph("mute=[10], strength=0.0", body), Paragraph("{10: 1.0}", body), Paragraph("{10: 1.0}", body), Paragraph("Pass", body_bold), Paragraph("BVA (lower bound)", body)],
        [Paragraph("TC02", body), Paragraph("build_scale_map", code_style), Paragraph("mute=[10], strength=1.0", body), Paragraph("{10: 0.0}", body), Paragraph("{10: 0.0}", body), Paragraph("Pass", body_bold), Paragraph("BVA (upper bound)", body)],
        [Paragraph("TC03", body), Paragraph("build_scale_map", code_style), Paragraph("mute=[100], s=0.3; boost=[200], s=0.5", body), Paragraph("{100: 0.7, 200: 1.5}", body), Paragraph("{100: 0.7, 200: 1.5}", body), Paragraph("Pass", body_bold), Paragraph("EP (disjoint sets)", body)],
        [Paragraph("TC04", body), Paragraph("build_scale_map", code_style), Paragraph("mute=[42], s=0.5; boost=[42], s=1.0", body), Paragraph("{42: 1.0} (0.5×2.0)", body), Paragraph("{42: 1.0}", body), Paragraph("Pass", body_bold), Paragraph("EP (overlap set)", body)],
        [Paragraph("TC05", body), Paragraph("with_extra_scale", code_style), Paragraph("base=None, fid=12, scale=0.8", body), Paragraph("{12: 0.8}", body), Paragraph("{12: 0.8}", body), Paragraph("Pass", body_bold), Paragraph("EP (empty base)", body)],
        [Paragraph("TC06", body), Paragraph("with_extra_scale", code_style), Paragraph("base={5: 2.0}, fid=5, scale=0.0", body), Paragraph("{5: 0.0}", body), Paragraph("{5: 0.0}", body), Paragraph("Pass", body_bold), Paragraph("BVA (zero scale)", body)],
        [Paragraph("TC07", body), Paragraph("evaluate_safety", code_style), Paragraph("r_k=8, r_0=8, delta=-1e-6", body), Paragraph("True (passes filter)", body), Paragraph("True", body), Paragraph("Pass", body_bold), Paragraph("BVA (boundary -1e-6)", body)],
        [Paragraph("TC08", body), Paragraph("evaluate_safety", code_style), Paragraph("r_k=8, r_0=8, delta=-1.01e-6", body), Paragraph("False (rejected)", body), Paragraph("False", body), Paragraph("Pass", body_bold), Paragraph("BVA (below threshold)", body)],
        [Paragraph("TC09", body), Paragraph("evaluate_safety", code_style), Paragraph("r_k=3, r_0=2, delta=+0.05", body), Paragraph("False (rank worsened)", body), Paragraph("False", body), Paragraph("Pass", body_bold), Paragraph("EP (rank regression)", body)],
        [Paragraph("TC10", body), Paragraph("_to_a (Eq. 3.5)", code_style), Paragraph("z = Z_ZERO (-0.693147)", body), Paragraph("a = 0.0 (multiplier=1.0)", body), Paragraph("a = 0.0", body), Paragraph("Pass", body_bold), Paragraph("BVA (zero-edit point)", body)],
        [Paragraph("TC11", body), Paragraph("_to_a (Eq. 3.5)", code_style), Paragraph("z = -100.0, z = +100.0", body), Paragraph("a = -1.0, a = +2.0", body), Paragraph("a = -1.0, a = +2.0", body), Paragraph("Pass", body_bold), Paragraph("BVA (asymptotic limits)", body)],
        [Paragraph("TC12", body), Paragraph("fmt_prob", code_style), Paragraph("p = 0.01 (1.0% threshold)", body), Paragraph("'1.00%' (no odds)", body), Paragraph("'1.00%'", body), Paragraph("Pass", body_bold), Paragraph("BVA (1% boundary)", body)],
        [Paragraph("TC13", body), Paragraph("fmt_prob", code_style), Paragraph("p = 0.0099 (just below 1%)", body), Paragraph("Contains '1 in 101'", body), Paragraph("'0.990% (1 in 101)'", body), Paragraph("Pass", body_bold), Paragraph("BVA (just below 1%)", body)],
        [Paragraph("TC14", body), Paragraph("fmt_prob", code_style), Paragraph("p = 0.0 and p = -0.05", body), Paragraph("'0' (non-positive)", body), Paragraph("'0'", body), Paragraph("Pass", body_bold), Paragraph("EP (non-positive input)", body)],
    ]

    tc_table = Table(tc_data, colWidths=[30, 95, 115, 95, 80, 40, 60])
    tc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(tc_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>(c) Testing Techniques:</b> <b>Boundary Value Analysis (BVA)</b> was used for TC01/TC02 (mute attenuation extremes 0.0 and 1.0), "
        "TC06 (zero scale multiplier), TC07/TC08 (safety tolerance boundary $\Delta p = -10^{-6}$ and its immediate violation), "
        "TC10/TC11 (gradient parameter zero-edit anchor $z=Z\_ZERO$ and asymptotic saturation bounds $m \in [0, 3]$), and "
        "TC12/TC13 (exact 1.0% odds threshold). <b>Equivalence Partitioning (EP)</b> was used for TC03 (disjoint partitions), "
        "TC04 (overlapping set partition), TC05 (uninitialized state partition), TC09 (invalid rank regression partition), and "
        "TC14 (non-positive probability domain).", body
    ))

    if os.path.exists("docs/assignment-2/evidence/terminal_tests.png"):
        story.append(Image("docs/assignment-2/evidence/terminal_tests.png", width=510, height=270))
        story.append(Paragraph("Figure 5: Terminal execution evidence: all 14 PyTest test cases passed in 2.85s", caption))

    story.append(PageBreak())

    # =========================================================================
    # QUESTION 4
    # =========================================================================
    story.append(Paragraph("Question - 4", q_title))
    story.append(Paragraph("Set up a CI/CD pipeline for your Main Project using GitHub Actions. [15 Marks]", q_prompt))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D3B66"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("<b>Answer</b>", body_bold))
    story.append(Paragraph(
        "<b>(a) GitHub Actions Workflow File:</b> The workflow file <code>.github/workflows/ci.yml</code> triggers automatically on every "
        "push and pull request targeting <code>se_assignment</code> and <code>feature/**</code> branches. It executes a test matrix across "
        "<b>Python 3.11</b> and <b>Python 3.12</b> on <code>ubuntu-latest</code>. To prevent CI timeouts, the pipeline installs the official "
        "CPU build of PyTorch (<code>--extra-index-url https://download.pytorch.org/whl/cpu</code>), installs PyTest, and executes "
        "<code>pytest -v tests/test_assignment_a2.py</code>, completing the entire validation matrix in under 45 seconds.", body
    ))

    if os.path.exists("docs/assignment-2/evidence/ci_pipeline.png"):
        story.append(Image("docs/assignment-2/evidence/ci_pipeline.png", width=510, height=260))
        story.append(Paragraph("Figure 6: GitHub Actions CI pipeline execution proof (Run #1: Success on Python 3.11 & 3.12)", caption))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>(b) Successful Pipeline Run & Workflow Status Badge:</b> GitHub Actions Run #1 executed on commit <code>2fbe101</code> "
        "pushed to <code>feature/alvin-app-testing</code> and completed with status <b>Success (green checkmark)</b> across all jobs. "
        "The live workflow status badge is active at the top of <code>README.md</code>:<br/>"
        "<code>[![CI](https://github.com/adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders/actions/workflows/ci.yml/badge.svg)](https://github.com/adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders/actions/workflows/ci.yml)</code>", body
    ))

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>(c) Cloud Deployment Note (Exact 5–6 Lines):</b>", body_bold))

    deploy_note = (
        "<b>IaaS fits best:</b> a virtual machine instance such as an AWS EC2 g4dn.xlarge (1 NVIDIA T4 GPU, 16 GB VRAM, 16 GB RAM) or an Azure NC4as T4 v3 "
        "running Ubuntu 22.04 LTS, providing dedicated CUDA hardware acceleration, persistent NVMe storage for large SAE dictionaries (24,576 latents per layer), "
        "and full control over GPU drivers and PyTorch environments.<br/>"
        "<b>PaaS</b> platforms (such as Streamlit Community Cloud or Google Cloud Run) are suitable for lightweight inference demos or the React frontend, "
        "but lack dedicated persistent GPU acceleration and hit memory ceilings during batched activation caching.<br/>"
        "<b>SaaS does not fit</b> because FeatureScalpel is an internal mechanistic interpretability research system that inspects intermediate network representations "
        "rather than exposing a turnkey multi-tenant consumer application.<br/>"
        "<b>Yes, FeatureScalpel can be packaged as a Docker container</b> using an NVIDIA CUDA base image (<code>nvidia/cuda:12.4.1-runtime-ubuntu22.04</code>) "
        "containing Python 3.11/3.12, PyTorch 2.6.0+cu124, TransformerLens, and SAELens, started via <code>docker compose up</code> with GPU pass-through (<code>--gpus all</code>) "
        "and health checks."
    )

    deploy_box = Table([[Paragraph(deploy_note, body)]], colWidths=[510])
    deploy_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
        ('BOX', (0, 0), (-1, -1), 1.2, colors.HexColor("#16A34A")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(deploy_box)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully compiled {PDF_FILENAME}")

if __name__ == "__main__":
    build_pdf()
