import os
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
        excel = ExcelData(test_input_file, normalisation= False, shuffle= False)
        test_output_file_1 = os.path.join(os.getcwd, f'test_normalisation.xlsx')
        excel.Normalisation()
        excel._data_frame.to_excel(test_output_file_1)
        excel.Save_Normalisation()
    
    #shuffle delle righe
    if test_shuffle:    
        excel = ExcelData(test_input_file, normalisation= False, shuffle= False)
        test_output_file_2 = os.path.join(os.getcwd, f'test_shuffle.xlsx')
        excel._data_frame.to_excel(test_output_file_2)

    #rimozione colonna
    if test_remove_column:    
        if remove_column_name == "" :
            remove_column_name = 'Colonna 1'
        excel = ExcelData(test_input_file, normalisation= False, shuffle= False)
        excel._remove_column(remove_column_name)
        print(excel.data_grid)

    #selezione colonna
    if test_select_column:    
        if select_column_name == "":
            select_column_name = 'Colonna 1'
        excel = ExcelData(test_input_file, normalisation= False, shuffle= False)
        print(excel.Select_column(select_column_name))
        
    #split del file
    if test_split == True:
        SplitExcel(test_input_file, 0)

if __name__ == 'main':
    #file input per test
    test_input_file = os.path.join(os.getcwd, f'test_input.xlsx')
    testExcelData(test_input_file,
                  test_normalisation = True,
                  test_shuffle = True,
                  test_remove_column = True,
                  test_select_column = True,
                  select_column_name = 'Colonna 2')