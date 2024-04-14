import pandas
import numpy

#input data from .xlsx
file1 = 'FS_features_ABIDE_males.xlsx'
df = pandas.read_excel(file1)

my_matrix = numpy.array(df.drop(columns=['FILE_ID', 'AGE_AT_SCAN']), float)
my_matrix_corrected = my_matrix.copy()

rows_to_keep = numpy.isnan(my_matrix).any(axis=1)
my_matrix_corrected = my_matrix[~rows_to_keep]

columns_means = numpy.mean(my_matrix_corrected[0])


covariance = numpy.cov(my_matrix_corrected - columns_means, rowvar = False)
eigenvalues, eigenvector = numpy.linalg.eig(covariance)


#print(df.head())
#print(my_matrix[0])
#print(my_matrix[13,2])
#print(numpy.abs(eigenvalues))