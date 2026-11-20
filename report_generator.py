"""Module 5 Support: Generate comprehensive local PDF reports using ReportLab.
Summarizes dataset properties, all 5 classification algorithms, metrics, and history.
"""
from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_pdf_report(bundle, prediction_rows=None):
    prediction_rows = prediction_rows or []
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D")
    )
    h2_style = ParagraphStyle(
        "CustomH2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0")
    )
    body_style = styles["Normal"]

    story = [
        Paragraph("SMS Spam Classifier — 5 Module Academic Report", title_style),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Local Execution (Zero External APIs)", body_style),
        Spacer(1, 10),
        
        Paragraph("Module 1 & 2: Dataset & Preprocessing Summary", h2_style),
        Paragraph(
            f"<b>Dataset:</b> {bundle.get('dataset_name', 'SMS Spam Dataset')} | "
            f"<b>Total Size:</b> {bundle.get('total_messages', len(bundle['df'])):,} rows<br/>"
            f"<b>Training Samples:</b> {bundle['training_samples']:,} | "
            f"<b>Testing Samples:</b> {bundle['testing_samples']:,}",
            body_style
        ),
        Paragraph(
            f"<b>Class Balance:</b> Ham = {bundle.get('ham_count', 0):,} "
            f"({(bundle.get('ham_count', 1) / max(bundle.get('total_messages', 1), 1)):.1%}) | "
            f"Spam = {bundle.get('spam_count', 0):,} "
            f"({(bundle.get('spam_count', 0) / max(bundle.get('total_messages', 1), 1)):.1%})",
            body_style
        ),
        Paragraph(
            "<b>NLP Preprocessing Pipeline:</b> URL/Phone/Currency regex tokenization, "
            "lowercasing, punctuation stripping, stopword filtering, and TF-IDF / Tokenizer sequence vectorization.",
            body_style
        ),
        Spacer(1, 10),
        
        Paragraph("Module 4 & 5: Multi-Algorithm Benchmark Comparison", h2_style),
    ]

    # Table with all 5 algorithms
    table_data = [["Algorithm", "Accuracy", "Spam Precision", "Spam Recall", "Spam F1", "ROC-AUC", "Train (s)"]]
    for name, m in bundle["metrics"].items():
        table_data.append([
            name,
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            f"{m.get('roc_auc', 0.0):.4f}",
            f"{m['training_time']:.2f}"
        ])

    table = Table(table_data, repeatRows=1, colWidths=[140, 60, 75, 65, 60, 60, 60])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F7FAFC"), colors.white]),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story += [
        Paragraph("Key Discriminative Spam Features (TF-IDF)", h2_style),
        Paragraph(", ".join(w for w, _ in bundle["spam_words"][:25]), body_style),
        Spacer(1, 10),
        
        Paragraph("Academic Evaluation & Conclusion", h2_style),
        Paragraph(
            "Due to natural class imbalance in SMS traffic (majority Ham), Spam F1-Score and ROC-AUC are "
            "the primary performance indicators rather than raw accuracy. Support Vector Machines and "
            "Bidirectional LSTMs consistently achieve strong precision and recall tradeoffs while running "
            "entirely offline without any cloud or API dependencies.",
            body_style
        ),
        Spacer(1, 8),
        Paragraph(f"Saved prediction records in user SQLite history: <b>{len(prediction_rows)}</b>", body_style)
    ]

    doc.build(story)
    return buffer.getvalue()
