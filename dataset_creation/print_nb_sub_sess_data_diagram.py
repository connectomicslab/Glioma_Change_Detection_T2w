import pandas as pd


def find_common_elements(list1: list, list2: list) -> list:
    """This function takes as input two lists and returns a list with the common elements
    Args:
        list1: first list
        list2: second list
    Returns:
        intersection_as_list: list containing the common elements between the two input lists
    """
    list1_as_set = set(list1)  # type: set
    intersection = list1_as_set.intersection(list2)  # type: set
    intersection_as_list = list(intersection)  # type: list

    return intersection_as_list


def extract_unique_elements(lst: list, ordered: bool = True) -> list:
    """This function extracts the unique elements of the input list (i.e. it removes duplicates)
    and returns them as an output list; if ordered=True (as by default), the returned list is ordered.
    Args:
        lst: input list from which we want to extract the unique elements
        ordered: whether the output list of unique values is sorted or not; True by default
    Returns:
        out_list: list containing unique values
    """
    out_list = list(set(lst))  # type: list

    if ordered:  # if we want to sort the list of unique values
        out_list.sort()  # type: list

    return out_list


def flatten_list(list_of_lists: list) -> list:
    """This function flattens the input list
    Args:
        list_of_lists: input list of lists that we want to flatten
    Returns:
        flattened_list: flattened list
    """
    flattened_list = [item for sublist in list_of_lists for item in sublist]

    return flattened_list


def find_unique_sub_ses(df, subs_to_exclude):
    all_subs = df["ipp"].values.tolist()
    all_subs_unique = extract_unique_elements(all_subs)
    all_sub_ses = []
    all_ses_diff = []
    total_nb_sessions = 0
    for sub in all_subs_unique:
        if sub not in subs_to_exclude:

            all_sess = []
            sub_ses = []
            ses_diff = []
            for idx, row in df.iterrows():
                if row["ipp"] == sub:
                    all_sess.append(row["exam_date"])
                    all_sess.append(row["comparative_date"])
                    sub_ses.append("{}_{}".format(row["ipp"], row["exam_date"]))
                    sub_ses.append("{}_{}".format(row["ipp"], row["comparative_date"]))
                    ses_diff.append("{}_{}_{}".format(row["ipp"], row["comparative_date"], row["exam_date"]))

            # all_sess_flat = flatten_list(all_sess)
            all_sess_unique = extract_unique_elements(all_sess)
            sub_ses_unique = extract_unique_elements(sub_ses)
            ses_diff_unique = extract_unique_elements(ses_diff)
            all_sub_ses.append(sub_ses_unique)
            all_ses_diff.append(ses_diff_unique)
            total_nb_sessions += len(all_sess_unique)

            # print("sub {}, {} sessions".format(sub, len(all_sess_unique)))

    return all_subs_unique, total_nb_sessions, flatten_list(all_sub_ses), flatten_list(all_ses_diff)


def find_sub_ses(df_all_wad, df_alex, df_chir, df_patric):

    sub_alex, nb_ses_alex, sub_ses_alex, ses_diff_alex = find_unique_sub_ses(df_alex, subs_to_exclude=[])
    sub_chir, nb_ses_chir, sub_ses_chir, ses_diff_chir = find_unique_sub_ses(df_chir, subs_to_exclude=[])
    sub_patric, nb_ses_patric, sub_ses_patric, ses_diff_patric = find_unique_sub_ses(df_patric, subs_to_exclude=[])

    all_subs_alex_and_chir = sub_alex + sub_chir
    all_subs_alex_and_chir_unique = extract_unique_elements(all_subs_alex_and_chir)

    intersect_patric_and_chiralex = find_common_elements(all_subs_alex_and_chir_unique, sub_patric)
    print("Unique subs tagged by Patric = {}; other subs belonging to same subs of Chir/Alex = {}".format(len(sub_patric) - len(intersect_patric_and_chiralex),
                                                                                                          len(intersect_patric_and_chiralex)))

    all_had_subs = sub_alex + sub_chir + sub_patric
    all_had_subs_unique = extract_unique_elements(all_had_subs)

    all_had_sub_ses = sub_ses_alex + sub_ses_chir + sub_ses_patric
    all_had_sub_ses_unique = extract_unique_elements(all_had_sub_ses)

    all_had_ses_diff = ses_diff_alex + ses_diff_chir + ses_diff_patric
    all_had_ses_diff_unique = extract_unique_elements(all_had_ses_diff)

    sub_wad, nb_ses_wad, sub_ses_wad, ses_diff_wad = find_unique_sub_ses(df_all_wad, subs_to_exclude=all_had_subs_unique)
    unique_subs_wad = [sub for sub in sub_wad if sub not in all_had_subs_unique]

    print("\nHAD dataset: {} unique subs, {} sess, {} sess pairs".format(len(all_had_subs_unique), len(all_had_sub_ses_unique), len(all_had_ses_diff_unique)))
    print("\nWAD dataset: {} unique subs (distinct from HAD), {} sess, {} sess pairs".format(len(unique_subs_wad), nb_ses_wad, len(ses_diff_wad)))
    print("\nTOT dataset: {} unique subs, {} sess, {} sess pairs".format(len(all_had_subs_unique) + len(unique_subs_wad),
                                                                         len(all_had_sub_ses_unique) + nb_ses_wad,
                                                                         len(all_had_ses_diff_unique) + len(ses_diff_wad)))


def main():
    all_wad_df_path = "/home/newuser/Desktop/Medical_Reports/df_comparative_dates_and_reports/df_dates_and_reports_third_prospective_batch_automatically_annotated_anonym_global_and_t2_labels_also_manual_subs_Mar_21_2022.csv"
    alex_df_path = "/home/newuser/Desktop/Medical_Reports/df_comparative_dates_and_reports/old_DO_NOT_DELETE_Jan_18_2022/Feb_11_2022_THESE_ARE_MOST_PROBABLY_ALL_CORRECT/df_dates_and_reports_alex_anonym_Feb_11_2022.csv"
    chir_df_path = "/home/newuser/Desktop/Medical_Reports/df_comparative_dates_and_reports/old_DO_NOT_DELETE_Jan_18_2022/Feb_11_2022_THESE_ARE_MOST_PROBABLY_ALL_CORRECT/df_dates_and_reports_chir_anonym_Feb_11_2022.csv"
    patric_df_path = "/home/newuser/Desktop/Medical_Reports/df_comparative_dates_and_reports/old_DO_NOT_DELETE_Jan_18_2022/Feb_11_2022_THESE_ARE_MOST_PROBABLY_ALL_CORRECT/df_dates_and_reports_patric_anonym_Feb_11_2022.csv"

    df_all_wad = pd.read_csv(all_wad_df_path)
    df_alex = pd.read_csv(alex_df_path)
    df_chir = pd.read_csv(chir_df_path)
    df_patric = pd.read_csv(patric_df_path)

    find_sub_ses(df_all_wad, df_alex, df_chir, df_patric)


if __name__ == '__main__':
    main()
