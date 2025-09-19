# ---------------------------------------------------------------------------------------------------------------------
#  Filename: util.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: Define assorted utilitarian functions.
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd


def print_fixed_width(list_of_strings):

    '''return a printable fixed-width table of 2 or 3 columns from a list of strings'''

    max_string_len = len(max(list_of_strings, key=len))

    # Guesstimate how much space columns will need. Goal is to minimize width.
    if max_string_len > 30:
        n_cols = 2
    elif max_string_len > 20:
        n_cols = 3
    else:
        n_cols = 4

    # Organize list elements alphabetically down the columns.
    list_of_strings = sorted(list_of_strings)

    # Determine how many rows are needed
    n_rows = int(np.ceil(len(list_of_strings) / n_cols))

    # Determine remainder after inserting data into columns
    remainder = (n_rows * n_cols) - len(list_of_strings)

    # Add blank list elements to completely fill columns
    list_of_strings.extend(['' for i in range(remainder)])

    # A list of lists, where each list contains the values for a column
    cols = [list_of_strings[(colnum * n_rows):(colnum * n_rows) + n_rows] for colnum in range(n_cols)]

    # Pandas to_string() method always justifies values to the right
    # (the 'justify' parameter is for headers only)
    # Left-justification is easier to read, but must be done manually.
    # Insert spaces to the right of each value to fill out the column's width.
    for i in range(len(cols)):
        col_list = cols[i]
        max_length = 0

        # Find the max string length in this column
        for this_value in col_list:
            max_length = max(max_length, len(this_value))

        # Iterate again and add blanks to reach the width
        new_values = []
        for this_value in col_list:
            n_spaces_to_add = max_length - len(this_value)
            spaces = ' ' * n_spaces_to_add
            new_values.append(this_value + spaces)

        cols[i] = new_values

    # Construct dataframe
    df = pd.DataFrame()
    for colnum in range(len(cols)):
        this_col = cols[colnum]
        df[f'{colnum}'] = this_col

    # Make all column headers blank, for printing to console.
    df.columns = [''] * n_cols

    return df.to_string(index=False)
