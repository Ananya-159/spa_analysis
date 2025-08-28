"""
Synthetic SPA Dataset Generator

Generates randomized student–project allocation (SPA) data for analysis and testing.
Outputs Excel data, metadata, and config snapshots to the /data/ folder.
This script is self-contained and does not use any external configuration file.
"""

import pandas as pd 
import numpy as np
import random
import os
import json
import math

def generate_dataset(num_students=102, num_projects=37, num_supervisors=21, seed=42):
    """
    Generating a synthetic SPA dataset with students, projects, and supervisors.
    Outputs an Excel file, summary text, and metadata/config files for reproducibility.
    """

    #  Configuration 
    random.seed(seed)  # For reproducibility
    np.random.seed(seed)
    max_supervisor_capacity = 6
    project_types = ['Research based', 'Client based', 'Student sourced']
    courses = ['Course 1', 'Course 2', 'Course 3']

    #  Supervisor Generation 
    supervisor_ids = [str(12762234 + i * 10000) for i in range(num_supervisors)]  # Unique numeric IDs
    supervisor_names = [chr(65 + i) for i in range(num_supervisors)]  # 'A' to 'U'
    num_projects_list = [2] * 16 + [1] * 5  # 16 supervisors get 2 projects, 5 get 1

    supervisors_df = pd.DataFrame({
        'Supervisor ID': supervisor_ids,
        'Supervisor Name': supervisor_names,
        'No. of Projects': num_projects_list
    })

    # Project Generation
    project_ids = [f'P{str(i+1).zfill(3)}' for i in range(num_projects)]  # P001, P002...
    project_titles = [f'Project {i+1}' for i in range(num_projects)]
    project_types_sampled = random.choices(project_types, k=num_projects)  # Random types assigned

    # Assigning the supervisors based on project count
    assigned_supervisors = []
    for i, count in enumerate(num_projects_list):
        assigned_supervisors.extend([supervisor_ids[i]] * count)  # Repeat ID based on project count

    min_students = [random.randint(1, 2) for _ in range(num_projects)]  # Random minimum students per project
    max_students = [random.randint(ms + 1, 5) for ms in min_students]  # Max is always greater than min
    avg_required = [random.randint(50, 75) for _ in range(num_projects)]  # Min GPA per project

    def generate_prereq():
        selected = random.sample(courses, k=random.randint(1, 2))  # 1 or 2 courses
        return ' | '.join([f"{course} ({random.choice([50, 60])})" for course in selected])  # With thresholds

    prerequisites = [generate_prereq() for _ in range(num_projects)]

    # Adding the second-round eligible flag (will be used later on)
    second_round_flags = [random.choice([True, False]) for _ in range(num_projects)]

    projects_df = pd.DataFrame({
        'Project ID': project_ids,
        'Project Title': project_titles,
        'Supervisor ID': assigned_supervisors,
        'Project Type': project_types_sampled,
        'Min Students': min_students,
        'Max Students': max_students,
        'Pre-requisites': prerequisites,
        'Minimum Average Required': avg_required,
        'Second Round Eligible': second_round_flags
    })

    #  Student Generation 
    student_ids = [str(9653507 + i) for i in range(num_students)]
    student_names = [f"S{i+1}" for i in range(num_students)]

    course_scores = np.random.randint(50, 85, size=(num_students, 3))  # Random marks between 50-84
    averages = course_scores.mean(axis=1).round().astype(int)  # GPA per student
    eligibility = [random.choice([True, False]) for _ in range(num_students)]  # 50/50 eligibility

    # Randomly pick 6 unique project IDs per student
    def generate_preferences():
        return random.sample(project_ids, 6)

    preferences = [generate_preferences() for _ in range(num_students)]

    students_df = pd.DataFrame({
        'Student ID': student_ids,
        'Student Name': student_names,
        'Course 1': course_scores[:, 0],
        'Course 2': course_scores[:, 1],
        'Course 3': course_scores[:, 2],
        'Average': averages,
        'Client Based Eligibility': eligibility
    })

    for i in range(6):
        students_df[f'Preference {i+1}'] = [prefs[i] for prefs in preferences]  # Creating columns Preference 1–6

    # Supervisor Capacity Calculation
    supervisor_project_cap = projects_df.groupby('Supervisor ID')['Max Students'].sum().reset_index()
    supervisor_project_cap.columns = ['Supervisor ID', 'Total Project Capacity']

    total_capacity = supervisor_project_cap['Total Project Capacity'].sum()
    supervisor_project_cap['Proportional Share'] = supervisor_project_cap['Total Project Capacity'] / total_capacity
    supervisor_project_cap['Max Student Capacity'] = (
        supervisor_project_cap['Proportional Share'] * num_students
    ).round().clip(upper=max_supervisor_capacity).astype(int)

    supervisor_project_cap['Min Student Capacity'] = (
        supervisor_project_cap['Max Student Capacity'] * 0.5
    ).apply(np.ceil).astype(int)

    supervisors_df = pd.merge(
        supervisors_df.drop(columns=['Min Student Capacity', 'Max Student Capacity'], errors='ignore'),
        supervisor_project_cap[['Supervisor ID', 'Min Student Capacity', 'Max Student Capacity']],
        on='Supervisor ID',
        how='left'
    )

    #  Validation 
    print(" Running dataset validation checks...")

    assert students_df['Student ID'].is_unique  # No duplicate students
    assert projects_df['Project ID'].is_unique  # No duplicate projects
    assert supervisors_df['Supervisor ID'].is_unique  # No duplicate supervisors

    project_id_set = set(projects_df['Project ID'])
    for i in range(1, 7):
        assert students_df[f'Preference {i}'].isin(project_id_set).all()  # All preferences are valid

    assert students_df['Average'].between(50, 100).all()  # GPA bounds
    assert (projects_df['Min Students'] <= projects_df['Max Students']).all()  # Min ≤ Max

    supervisor_project_counts = projects_df['Supervisor ID'].value_counts().to_dict()
    declared_project_counts = supervisors_df.set_index('Supervisor ID')['No. of Projects'].to_dict()
    for sid, expected in declared_project_counts.items():
        actual = supervisor_project_counts.get(sid, 0)
        assert actual == expected  # Supervisor's declared vs actual projects match

    for df_name, df in [('Students', students_df), ('Projects', projects_df), ('Supervisors', supervisors_df)]:
        assert not df.isnull().any().any(), f"Null values found in {df_name}"

    print(" All checks passed.\n")

    # Dataset Summary
    print(" Dataset Summary:\n")

    mean_gpa = students_df["Average"].mean().round(2)  # Average GPA
    eligible_pct = students_df["Client Based Eligibility"].mean() * 100  # % eligible for client projects
    top3_prefs = students_df[['Preference 1', 'Preference 2', 'Preference 3']]
    unique_projects_in_top3 = top3_prefs.values.flatten()
    project_overlap_rate = len(set(unique_projects_in_top3)) / len(projects_df) * 100   # % of projects appearing in at least one of the student's top 3 preferences
    min_cap = projects_df["Min Students"].min()
    max_cap = projects_df["Max Students"].max()
    supervisor_cap_max = supervisors_df["Max Student Capacity"].max()
    supervisor_cap_min = supervisors_df["Min Student Capacity"].min()

    summary = (
        f"- Total Students: {len(students_df)}\n"
        f"- Total Projects: {len(projects_df)}\n"
        f"- Total Supervisors: {len(supervisors_df)}\n"
        f"- Mean GPA: {mean_gpa}\n"
        f"- Client Eligible Students: {eligible_pct:.1f}%\n"
        f"- Preference overlap (top 3): {project_overlap_rate:.1f}% of projects\n"
        f"- Project Min Capacity: {min_cap}\n"
        f"- Project Max Capacity: {max_cap}\n"
        f"- Supervisor Max Student Capacity: {supervisor_cap_max}\n"
        f"- Supervisor Min Student Capacity: {supervisor_cap_min}"
    )

    print(summary)

    # Saving the Files 
    output_dir = os.path.join("..", "data")  # Save to parent/data directory
    os.makedirs(output_dir, exist_ok=True)

    excel_path = os.path.join(output_dir, "SPA_Dataset_With_Min_Max_Capacity.xlsx")
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        students_df.to_excel(writer, sheet_name="Students", index=False)
        projects_df.to_excel(writer, sheet_name="Projects", index=False)
        supervisors_df.to_excel(writer, sheet_name="Supervisors", index=False)

    with open(os.path.join(output_dir, "dataset_summary.txt"), "w") as f:
        f.write(summary)

    config = {
        "random_seed": seed,
        "num_students": num_students,
        "num_projects": num_projects,
        "num_supervisors": num_supervisors,
        "max_supervisor_capacity": max_supervisor_capacity,
        "num_preferences": 6,
        "num_courses": 3,
        "project_types": project_types,
        "course_names": courses,
        "min_students_range": [1, 2],
        "max_students_range": [2, 5],
        "average_required_range": [50, 75],
        "client_eligibility_ratio": "random ~50%",
        "use_second_round_flag": True
    }

    with open(os.path.join(output_dir, "dataset_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Metadata 
    metadata = {
        "Student ID": "Unique ID for each student",
        "Student Name": "Auto-generated label (e.g., S1, S2...)",
        "Course 1": "Simulated course score (50-84)",
        "Course 2": "Simulated course score (50-84)",
        "Course 3": "Simulated course score (50-84)",
        "Average": "GPA from 3 simulated course scores",
        "Client Based Eligibility": "Boolean for eligibility to client-based projects",
        "Preference 1-6": "Student's ranked project preferences",
        "Project ID": "Unique ID for each project",
        "Project Title": "Name of the project (e.g., Project 1)",
        "Project Type": "Type of project: Research, Client-based, or Student-sourced",
        "Min Students": "Minimum required students per project",
        "Max Students": "Maximum allowed students per project",
        "Pre-requisites": "Course requirements for the project with score thresholds",
        "Minimum Average Required": "Minimum GPA required to be eligible for a project",
        "Second Round Eligible": "Project can be reused in second round allocation",
        "Supervisor ID": "ID of the supervisor offering the project",
        "Supervisor Name": "Auto-generated name (A-U) for supervisor identity",
        "No. of Projects": "Number of projects offered by the supervisor",
        "Min Student Capacity": "Minimum number of students assigned to supervisor",
        "Max Student Capacity": "Maximum number of students allowed per supervisor"
    }

    with open(os.path.join(output_dir, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n Dataset files saved to: {output_dir}")

#  Running the script
if __name__ == "__main__":
    generate_dataset()
