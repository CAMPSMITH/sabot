def make_label(row, current_price_col, future_price_col, offset_col):
    if row[current_price_col] - row[future_price_col] > row[offset_col]:
        return -1
    elif row[future_price_col] - row[current_price_col] > row[offset_col]:
        return 1
    else:
        return 0