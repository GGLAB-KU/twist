import pandas as pd
import glob


def merge_csv_files(university, subject):
    # Define the file pattern for the CSV files
    file_pattern = f"{university}_{subject}_page_*.csv"
    csv_files = glob.glob(file_pattern)

    # List to hold dataframes
    dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate all the dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save the combined dataframe to a new CSV file
    combined_csv_filename = f"{university}_{subject}.csv"
    combined_df.to_csv(combined_csv_filename, index=False)

    print(f"All pages merged into {combined_csv_filename}")



if __name__ == "__main__":
    universities = [
        "Boğaziçi Üniversitesi",
        "Koç Üniversitesi",
        "İhsan Doğramacı Bilkent Üniversitesi",
        "Sabancı Üniversitesi",
        "Orta Doğu Teknik Üniversitesi",
        "İstanbul Teknik Üniversitesi"
    ]

    subjects = [
        "Bilgisayar Mühendisliği Bilimleri-Bilgisayar ve Kontrol = Computer Engineering and Computer Science and Control",
        "Matematik = Mathematics",
        "Fizik ve Fizik Mühendisliği = Physics and Physics Engineering",
        "Elektrik ve Elektronik Mühendisliği = Electrical and Electronics Engineering",
        "Endüstri ve Endüstri Mühendisliği = Industrial and Industrial Engineering",
        "Makine Mühendisliği = Mechanical Engineering"
    ]

    university = universities[4]
    subject = subjects[5]

    merge_csv_files(university, subject)

