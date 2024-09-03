import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd

from ExcelData import ExcelData
from SplitExcel import SplitExcel

def testExcelData(test_input_file,
                  test_normalisation = False,
                  test_shuffle = False,
                  test_remove_column = False,
                  remove_column_name = "",
                  test_select_column = False,
                  select_column_name = "",
                  test_split = False
                  ):
    
    #normalizzazione del file
    if test_normalisation:
        print("test normalisation start")
        excel = ExcelData(test_input_file, normalisation= False, shuffle= False)
        test_output_file_1 = os.path.join(os.getcwd(), f'test_normalisation.xlsx')
        excel.Normalisation()
        excel._data_frame.to_excel(test_output_file_1)
        print(excel.data_grid)
        excel.Save_Normalisation()
        print("test normalisation complete")

    
    #shuffle delle righe
    if test_shuffle:    
        print("test shuffle start")
        excel = ExcelData(test_input_file, normalisation= False, shuffle= False)
        test_output_file_2 = os.path.join(os.getcwd(), f'test_shuffle.xlsx')
        excel._data_frame.to_excel(test_output_file_2)
        print("test shuffle complete")

    #rimozione colonna
    if test_remove_column:    
        print("test remove start")
        if remove_column_name == "" :
            remove_column_name = 'FIQ'
        excel = ExcelData(test_input_file, normalisation= False, shuffle= False)
        test_output_file_3 = os.path.join(os.getcwd(), f'test_remove_column.xlsx')
        excel._remove_column(remove_column_name)
        excel._data_frame.to_excel(test_output_file_3)
        print(excel.data_grid)
        print("test remove complete")

    #selezione colonna
    if test_select_column:    
        print("test select start")
        if select_column_name == "":
            select_column_name = 'AGE_AT_SCAN'
        excel = ExcelData(test_input_file, normalisation= False, shuffle= False)
        test_output_file_4 = os.path.join(os.getcwd(), f'test_select_column.xlsx')
        excel._data_frame.to_excel(test_output_file_4)
        print(excel.Select_column(select_column_name))
        print("test select complete")
        
    #split del file
    if test_split == True:
        print("test split start")
        SplitExcel(test_input_file, 0)
        print("test split complete")

if __name__ == '__main__':
    #file input per test
    test_input_file = os.path.join(os.getcwd(), f'test_input.xlsx')
    testExcelData(test_input_file,
                  test_normalisation = True,
                  test_shuffle = True,
                  test_remove_column = True,
                  test_select_column = True,
                  select_column_name = 'FIQ',
                  test_split = True)