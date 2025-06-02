def get_config(file_name):
    if file_name == "mag_eng":
        return 10000, 10000, 0.5, 0.4
    elif file_name == "mag_cs":
        return 20000, 20000, 0.3, 0.4
    elif file_name == "mag_chem":
        return 20000, 20000, 0.6, 0.5
    elif file_name == "mag_med":
        return 20000, 20000, 0.2, 0.4
    elif file_name == "fb_348":
        return 500, 500, 0.2, 0.2
    elif file_name == "fb_414":
        return 500, 500, 0.3, 0.3
    elif file_name == "fb_686":
        return 400, 400, 0.3, 0.2
    elif file_name == "fb_698":
        return 4000, 4000, 0.2, 0.7
    elif file_name == "fb_1912":
        return 5000, 5000, 0.2, 0.3