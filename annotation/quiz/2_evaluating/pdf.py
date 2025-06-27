import pandas as pd
from fpdf import FPDF



df_all = pd.read_excel('results/df_all.xlsx')

ANNOTATOR_MAILS = ['tgbyzt@gmail.com', 'thehalil.donmez@gmail.com', 'yigitcankarakas@gmail.com']
GROUND_TRUTH_MAIL = 'gosahin@ku.edu.tr'




# Add introductory code section
sections = [
    ("1. English Term Detection",
     "- True Positives (TP): The total number of words correctly identified as part of English terms. For example, if 'machine learning' is detected, the TP count is 2 (one for each word).\n"
     "- False Positives (FP): The total number of words incorrectly identified as part of English terms. For example, if 'caesar salad' is detected but is not a relevant term, the FP count is 2.\n"
     "- False Negatives (FN): The total number of words that should have been identified as part of English terms but were missed. For example, if 'machine learning' is not detected, the FN count is 2.\n"
     "- True Negatives (TN): The total number of words correctly identified as not being part of English terms. For example, if 'caesar salad' is not detected and it is not a relevant term, the TN count is 2.\n"
     "- Precision: The ratio of correctly identified English term words (TP) to the total number of words identified as English terms (TP + FP). Precision = TP / (TP + FP).\n"
     "- Recall: The ratio of correctly identified English term words (TP) to the total number of words that should have been identified as English terms (TP + FN). Recall = TP / (TP + FN).\n"
     "- F1 Score: The harmonic mean of precision and recall. F1 Score = 2 * (Precision * Recall) / (Precision + Recall).\n"
     "- Accuracy: The ratio of correctly identified English term words (TP + TN) to the total number of words. Accuracy = (TP + TN) / (TP + FP + FN + TN).\n"),

    ("2. Turkish Term Detection",
     "- True Positives (TP): The total number of words correctly identified as part of Turkish terms. For example, if 'makine öğrenimi' is detected, the TP count is 2 (one for each word).\n"
     "- False Positives (FP): The total number of words incorrectly identified as part of Turkish terms. For example, if 'sezar salatası' is detected but is not a relevant term, the FP count is 2.\n"
     "- False Negatives (FN): The total number of words that should have been identified as part of Turkish terms but were missed. For example, if 'makine öğrenimi' is not detected, the FN count is 2.\n"
     "- True Negatives (TN): The total number of words correctly identified as not being part of Turkish terms. For example, if 'sezar salatası' is not detected and it is not a relevant term, the TN count is 2.\n"
     "- Precision: The ratio of correctly identified Turkish term words (TP) to the total number of words identified as Turkish terms (TP + FP). Precision = TP / (TP + FP).\n"
     "- Recall: The ratio of correctly identified Turkish term words (TP) to the total number of words that should have been identified as Turkish terms (TP + FN). Recall = TP / (TP + FN).\n"
     "- F1 Score: The harmonic mean of precision and recall. F1 Score = 2 * (Precision * Recall) / (Precision + Recall).\n"
     "- Accuracy: The ratio of correctly identified Turkish term words (TP + TN) to the total number of words. Accuracy = (TP + TN) / (TP + FP + FN + TN).\n"),

    ("3. Turkish Labels Detection",
     "- Intersection: The total number of terms that are correctly identified with the true labels. For example, if 'makine öğrenimi' is labeled as CORRECT_TRANSLATION, the intersection count is 1.\n"
     "Note: Unlike English and Turkish Term Detection, where the focus is on detecting individual words, Turkish Labels Detection focuses on entire terms. Also, only the terms present in the ground truth are considered, meaning Turkish Labels Detection is a subset of Turkish Term Detection.\n"
     "- Difference: The total number of terms that are incorrectly identified with the true labels. For example, if 'makine öğrenimi' is labeled as WRONG_TRANSLATION, the difference count is 1.\n"
     "- Exact Match: The ratio of correctly identified terms (Intersection) to the total number of terms. Exact Match = Intersection / (Intersection + Difference).\n"),

    ("4. Correction",
     "- Intersection: The total number of terms that are rectified correctly. For example, if 'makina öğrenmesi' is corrected as 'makine öğrenimi', the intersection count is 1.\n"
     "- Difference: The total number of terms that are rectified incorrectly. For example, if 'makina öğrenmesi' is corrected as 'makine öğrenmesi', the difference count is 1.\n"
     "- Exact Match: The ratio of correctly rectified terms (Intersection) to the total number of terms. Exact Match = Intersection / (Intersection + Difference).\n"),

    ("5. Term Linking",
     "- Intersection: The total number of terms that are correctly linked to the true English terms with the help of terimler.org. For example, if 'machine learning' is linked to 'https://terimler.org/terim/makine-ogrenimi', the intersection count is 1.\n"
     "- Difference: The total number of terms that are incorrectly linked to the true English terms. For example, if 'machine learning' is linked to 'https://terimler.org/terim/sirali-ogrenme', the difference count is 1.\n"
     "- Exact Match: The ratio of correctly linked terms (Intersection) to the total number of terms. Exact Match = Intersection / (Intersection + Difference).\n")
]



# Create a PDF class with color support
class PDF(FPDF):
    def header(self):
        self.set_font('DejaVu', 'B', 16)
        self.set_text_color(0, 0, 128)  # Dark blue color for the header
        self.cell(0, 10, '', 0, 1, 'C')
        self.ln(10)

    def add_thank_you_page(self, mail, name):
        self.add_page()
        self.set_font('DejaVu', 'B', 18)
        self.set_text_color(0, 100, 0)  # Green color for the thank you header
        self.cell(0, 10, 'Quiz Evaluation Report', 0, 1, 'C')
        self.ln(20)
        self.set_font('DejaVu', '', 12)
        self.set_text_color(0, 0, 0)  # Black color for the body text

        body = f"Sayın {name}"
        body += " (" + mail + ")"
        body += ",\n\n"

        body += "Quiz'deki başarınızdan dolayı sizi kutluyoruz! Bu rapor, her bir görevdeki performansınızın ayrıntılı bir analizini sunarak, gerçek etiketlemede size yardımcı olmayı ve aynı hataları tekrarlamaktan kaçınmanızı sağlamayı amaçlamaktadır. Öncelikle, değerlendirmede kullanılan metriklerin açıklamalarını sunuyoruz, ardından her bir görevdeki performansınızın ayrıntılı bir dökümünü sağlıyoruz. Raporun sonunda, tüm görevlerdeki performansınızın toplu bir özetini sunuyoruz. Bu rapor, quiz etiketlemelerinize dayanarak otomatik olarak oluşturulmuştur. Not: Bir terimin ne olduğuna dair kesin bir tanım yoktur. Bu nedenle, bu rapordaki \"ground truth\" ekibimizin görüşlerini yansıtmaktadır ve tamamen nesnel bir değerlendirme değildir."
        self.multi_cell(0, 10, body)
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 14)
        self.set_text_color(0, 0, 128)  # Dark blue color for the chapter title
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('DejaVu', '', 12)
        self.set_text_color(0, 0, 0)  # Black color for the body text
        self.multi_cell(0, 10, body)
        self.ln()

    def add_chapter(self, title, body):
        self.add_page()
        self.chapter_title(title)

        # Process the body text and bold the text between "-" and ":"
        lines = body.split('\n')
        for line in lines:
            if "- " in line and ": " in line:
                # Find the position of "-" and ":"
                dash_index = line.find("- ")
                colon_index = line.find(": ")

                # Extract the text before, between, and after "-" and ":"
                before_dash = line[1:dash_index + 2]
                bold_text = line[dash_index + 2:colon_index + 2]
                after_colon = line[colon_index + 2:]

                # Print the text before "-", the bold text, and the text after ":"
                # set text color to black
                self.set_text_color(0, 0, 0)
                self.set_font('DejaVu', '', 12)
                self.multi_cell(0, 10, before_dash)
                self.set_font('DejaVu', 'B', 12)
                self.multi_cell(0, 10, bold_text)
                self.set_font('DejaVu', '', 12)
                self.multi_cell(0, 10, after_colon)
            else:
                # For lines that don't need bolding, print as usual
                self.multi_cell(0, 10, line)



    def add_bold_text(self, text):
        self.set_font('DejaVu', 'B', 12)
        self.set_text_color(0, 0, 128)  # Dark blue color for bold text
        self.cell(0, 10, text, 0, 1, 'L')
        self.set_font('DejaVu', '', 12)


def generate_task_page(pdf, task_number, df_all, annotator_mail):
    if task_number == 1:
        pdf.add_page()
    pdf.add_bold_text(f"TASK-{task_number} Performance Report:")
    # Add the task-specific screenshot
    screenshot_path = f'png/{task_number}.png'
    pdf.image(screenshot_path, x=10, w=190)  # Adjust the x and w as needed for alignment and size
    pdf.ln(10)  # Space after the image


    task_prefix = f'task_{task_number}'

    # Filter the relevant columns for the specific task
    task_columns = [col for col in df_all.columns if col.startswith(task_prefix)]

    # Extract data for the specific annotator and ground truth
    annotator_data = df_all.loc[df_all['mail'] == annotator_mail, ['mail'] + task_columns].squeeze()
    ground_truth_data = df_all.loc[df_all['mail'] == GROUND_TRUTH_MAIL, task_columns].squeeze()

    if annotator_data.empty:
        return f"No data found for {annotator_mail} on Task {task_number}."

    if ground_truth_data.empty:
        return f"No ground truth data found for {GROUND_TRUTH_MAIL} on Task {task_number}."




    # 1. English Term Detection
    pdf.add_bold_text("1. English Term Detection:")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10, f"  - Number of TP: {annotator_data.get(f'{task_prefix}_english_term_detection_tp', 'N/A')}")
    pdf.multi_cell(0, 10, f"  - Number of FP: {annotator_data.get(f'{task_prefix}_english_term_detection_fp', 'N/A')}")
    fp_set = eval(annotator_data.get(f'{task_prefix}_english_term_detection_fp_set', '[]'))
    fp_set = eval(str(annotator_data.get(f'{task_prefix}_english_term_detection_fp_set', '[]')))

    for word, _, _, sentence_index in fp_set:
        pdf.set_text_color(255, 0, 0)  # Red for incorrectly labeled terms
        pdf.cell(10)
        pdf.multi_cell(0, 10, f"    * \"{word}\" in {sentence_index} is incorrectly labeled as a term.")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10, f"  - Number of FN: {annotator_data.get(f'{task_prefix}_english_term_detection_fn', 'N/A')}")
    fn_set = eval(annotator_data.get(f'{task_prefix}_english_term_detection_fn_set', '[]'))
    for word, _, _, sentence_index in fn_set:
        pdf.set_text_color(255, 0, 0)  # Red for incorrectly labeled terms
        pdf.cell(10)
        pdf.multi_cell(0, 10, f"    * \"{word}\" is a term in {sentence_index} but you missed it.")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10, f"  - Precision: {annotator_data.get(f'{task_prefix}_english_term_detection_precision', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - Recall: {annotator_data.get(f'{task_prefix}_english_term_detection_recall', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - F1 Score: {annotator_data.get(f'{task_prefix}_english_term_detection_f1_score', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - Accuracy: {annotator_data.get(f'{task_prefix}_english_term_detection_accuracy', 'N/A'):.2%}")
    pdf.ln(5)


    # 2. Turkish Term Detection
    pdf.add_bold_text("2. Turkish Term Detection:")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10, f"  - Number of TP: {annotator_data.get(f'{task_prefix}_turkish_term_detection_tp', 'N/A')}")
    pdf.multi_cell(0, 10, f"  - Number of FP: {annotator_data.get(f'{task_prefix}_turkish_term_detection_fp', 'N/A')}")
    fp_set = eval(annotator_data.get(f'{task_prefix}_turkish_term_detection_fp_set', '[]'))
    for word, _, _, sentence_index in fp_set:
        pdf.cell(10)
        # set color to turquoise for the Turkish term detection
        pdf.set_text_color(255, 165, 0)
        pdf.multi_cell(0, 10, f"    * \"{word}\" in {sentence_index} is incorrectly labeled as a term.")

    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10, f"  - Number of FN: {annotator_data.get(f'{task_prefix}_turkish_term_detection_fn', 'N/A')}")
    fn_set = eval(annotator_data.get(f'{task_prefix}_turkish_term_detection_fn_set', '[]'))
    for word, _, _, sentence_index in fn_set:
        pdf.cell(10)
        # set color to turquoise for the Turkish term detection
        pdf.set_text_color(255, 165, 0)
        pdf.multi_cell(0, 10, f"    * \"{word}\" is a term in {sentence_index} but you missed it.")

    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10, f"  - Precision: {annotator_data.get(f'{task_prefix}_turkish_term_detection_precision', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - Recall: {annotator_data.get(f'{task_prefix}_turkish_term_detection_recall', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - F1 Score: {annotator_data.get(f'{task_prefix}_turkish_term_detection_f1_score', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - Accuracy: {annotator_data.get(f'{task_prefix}_turkish_term_detection_accuracy', 'N/A'):.2%}")
    pdf.ln(5)


    # 3. Turkish Labels Detection
    pdf.add_bold_text("3. Turkish Labels Detection:")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10, f"  - Number of Intersection: {annotator_data.get(f'{task_prefix}_turkish_labels_intersection_num', 'N/A')}")
    pdf.multi_cell(0, 10, f"  - Number of Difference: {annotator_data.get(f'{task_prefix}_turkish_labels_difference_num', 'N/A')}")
    pdf.multi_cell(0, 10, f"  - Exact Match: {annotator_data.get(f'{task_prefix}_turkish_labels_exact_match', 'N/A'):.2%}")

    difference_set = eval(annotator_data.get(f'{task_prefix}_turkish_labels_difference_set', '[]'))
    ground_truth_label_set = eval(ground_truth_data.get(f'{task_prefix}_turkish_labels_intersection_set', '[]'))

    same_indices = []
    for i, (word, start, end, sentence_index, label) in enumerate(difference_set):
        for j, (word_gt, start_gt, end_gt, sentence_index_gt, label_gt) in enumerate(ground_truth_label_set):
            if start == start_gt and end == end_gt:
                same_indices.append((i, j))
                break

    same_indices = list(set(same_indices))
    difference_set = list(difference_set)
    ground_truth_label_set = list(ground_truth_label_set)

    for i, j in same_indices:
        word, start, end, sentence_index, label = difference_set[i]
        word_gt, start_gt, end_gt, sentence_index_gt, label_gt = ground_truth_label_set[j]
        if label != label_gt:
            pdf.cell(10)
            # set text to pink for the Turkish labels detection
            pdf.set_text_color(255, 20, 147)
            pdf.multi_cell(0, 10, f"    * \"{word}\" is labeled as \"{label}\" in {sentence_index} but it should be \"{label_gt}\".")
    pdf.ln(5)

    # Reset the text color to black
    pdf.set_text_color(0, 0, 0)


    # 4. Turkish Translation Corrections
    pdf.add_bold_text("4. Turkish Translation Corrections:")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10, f"  - Number of Intersection: {annotator_data.get(f'{task_prefix}_turkish_corrections_intersection_num', 'N/A')}")
    pdf.multi_cell(0, 10, f"  - Number of Difference: {annotator_data.get(f'{task_prefix}_turkish_corrections_difference_num', 'N/A')}")
    pdf.multi_cell(0, 10, f"  - Exact Match: {annotator_data.get(f'{task_prefix}_turkish_corrections_exact_match', 'N/A'):.2%}")


    difference_set_turkish_corrections = eval(annotator_data.get(f'{task_prefix}_turkish_corrections_difference_set', '[]'))
    ground_truth_correction_set = eval(ground_truth_data.get(f'{task_prefix}_turkish_corrections_intersection_set', '[]'))

    same_indices = []
    for i, (word, start, end, sentence_index, label) in enumerate(difference_set_turkish_corrections):
        for j, (word_gt, start_gt, end_gt, sentence_index_gt, label_gt) in enumerate(ground_truth_correction_set):
            if start == start_gt and end == end_gt:
                same_indices.append((i, j))
                break

    same_indices = list(set(same_indices))
    difference_set_turkish_corrections = list(difference_set_turkish_corrections)
    ground_truth_correction_set = list(ground_truth_correction_set)

    for i, j in same_indices:
        word, start, end, sentence_index, label = difference_set_turkish_corrections[i]
        word_gt, start_gt, end_gt, sentence_index_gt, label_gt = ground_truth_correction_set[j]
        if label != label_gt:
            pdf.cell(10)
            # set text color to purple for the Turkish translation corrections
            pdf.set_text_color(128, 0, 128)
            pdf.multi_cell(0, 10, f"    * \"{word}\" is rectified as \"{label}\" in {sentence_index} but it should be \"{label_gt}\".")

    # Reset the text color to black
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # 5. English Term Linking
    pdf.add_bold_text("5. Term Linking:")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10, f"  - Number of TP: {annotator_data.get(f'{task_prefix}_english_term_linking_intersection_num', 'N/A')}")
    pdf.multi_cell(0, 10, f"  - Number of Difference: {annotator_data.get(f'{task_prefix}_english_term_linking_difference_num', 'N/A')}")
    pdf.multi_cell(0, 10, f"  - Exact Match: {annotator_data.get(f'{task_prefix}_english_term_linking_exact_match', 'N/A'):.2%}")

    difference_set_english_term_linking = eval(annotator_data.get(f'{task_prefix}_english_term_linking_difference_set', '[]'))
    ground_truth_linking_set = eval(ground_truth_data.get(f'{task_prefix}_english_term_linking_intersection_set', '[]'))

    same_indices = []
    for i, (word, start, end, sentence_index, label) in enumerate(difference_set_english_term_linking):
        for j, (word_gt, start_gt, end_gt, sentence_index_gt, label_gt) in enumerate(ground_truth_linking_set):
            if start == start_gt and end == end_gt:
                same_indices.append((i, j))
                break

    same_indices = list(set(same_indices))
    difference_set_english_term_linking = list(difference_set_english_term_linking)
    ground_truth_linking_set = list(ground_truth_linking_set)

    for i, j in same_indices:
        word, start, end, sentence_index, label = difference_set_english_term_linking[i]
        word_gt, start_gt, end_gt, sentence_index_gt, label_gt = ground_truth_linking_set[j]
        if label != label_gt:
            pdf.cell(10)
            # set text color to orange for the English term linking
            pdf.set_text_color(64, 224, 208)
            pdf.multi_cell(0, 10, f"    * \"{word}\" is linked as \"{label}\" in {sentence_index} but it should be \"{label_gt}\".")

    # Reset the text color to black
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    pdf.add_page()


for mail in ANNOTATOR_MAILS:
    pdf = PDF()

    # Add DejaVu fonts to the PDF
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)

    name = df_all.loc[df_all['mail'] == mail, 'AdSoyad'].values[0]

    # Add the "Thank You" page with the screenshot
    pdf.add_thank_you_page(mail, name)

    for title, body in sections:
        pdf.add_chapter(title, body)
        pdf.ln(10)

    number_of_tasks_completed = df_all.loc[df_all['mail'] == mail, 'number_of_tasks_completed'].values[0]

    completed_tasks = []
    for i in range(1, 11):  # Assuming tasks are numbered 1 through 10
        task_prefix = f'task_{i}'
        if any(col.startswith(task_prefix) and not df_all.loc[df_all['mail'] == mail, col].isna().all() for col in
               df_all.columns):
            completed_tasks.append(i)

    for task_number in completed_tasks:
        generate_task_page(pdf, task_number, df_all, mail)

    # Add the summary page
    pdf.chapter_title("Summary of All Tasks")
    pdf.chapter_body(
        f"Here is the summary of your performance on all tasks. For each task, the metrics include precision, recall, F1 score, and accuracy, and the exact match where applicable. ")

    relevant_columns = [
        "cumulative_english_term_detection_precision",
        "cumulative_english_term_detection_recall",
        "cumulative_english_term_detection_f1_score",
        "cumulative_english_term_detection_accuracy",
        "cumulative_turkish_term_detection_precision",
        "cumulative_turkish_term_detection_recall",
        "cumulative_turkish_term_detection_f1_score",
        "cumulative_turkish_term_detection_accuracy",
        "cumulative_turkish_labels_exact_match",
        "cumulative_turkish_corrections_exact_match",
        "cumulative_english_term_linking_exact_match"
    ]

    summary_data = df_all.loc[df_all['mail'] == mail, relevant_columns].squeeze()

    # 1. English Term Detection
    pdf.add_bold_text("1. Cumulative English Term Detection:")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10,
                   f"  - Precision: {summary_data.get('cumulative_english_term_detection_precision', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - Recall: {summary_data.get('cumulative_english_term_detection_recall', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - F1 Score: {summary_data.get('cumulative_english_term_detection_f1_score', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - Accuracy: {summary_data.get('cumulative_english_term_detection_accuracy', 'N/A'):.2%}")
    pdf.ln(5)

    # 2. Turkish Term Detection
    pdf.add_bold_text("2. Cumulative Turkish Term Detection:")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10,
                   f"  - Precision: {summary_data.get('cumulative_turkish_term_detection_precision', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - Recall: {summary_data.get('cumulative_turkish_term_detection_recall', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - F1 Score: {summary_data.get('cumulative_turkish_term_detection_f1_score', 'N/A'):.2%}")
    pdf.multi_cell(0, 10, f"  - Accuracy: {summary_data.get('cumulative_turkish_term_detection_accuracy', 'N/A'):.2%}")
    pdf.ln(5)

    # 3. Turkish Labels Detection
    pdf.add_bold_text("3. Cumulative Turkish Labels Detection:")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10, f"  - Exact Match: {summary_data.get('cumulative_turkish_labels_exact_match', 'N/A'):.2%}")
    pdf.ln(5)

    # 4. Turkish Translation Corrections
    pdf.add_bold_text("4. Cumulative Turkish Translation Corrections:")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10,
                   f"  - Exact Match: {summary_data.get('cumulative_turkish_corrections_exact_match', 'N/A'):.2%}")
    pdf.ln(5)

    # 5. English Term Linking
    pdf.add_bold_text("5. Cumulative Term Linking:")
    pdf.set_text_color(0, 0, 0)  # Black for regular text
    pdf.multi_cell(0, 10,
                   f"  - Exact Match: {summary_data.get('cumulative_english_term_linking_exact_match', 'N/A'):.2%}")
    pdf.ln(10)

    # Clean up the email to create a valid filename
    valid_email = mail.strip().replace('@', '_at_').replace('.', '_')

    # Generate a clean PDF file name
    pdf_file_name = f'{valid_email}_quiz_evaluation_report.pdf'

    # Save the PDF file
    pdf.output('pdf/'+pdf_file_name)

    print(f'PDF report created: {pdf_file_name}')
