using XLSX, DataFrames
using Plots

df = DataFrame(XLSX.readtable("ControlError_Numbers.xlsx", "Sheet1"))

df.Mean = float.(df.Mean)
df.STD = float.(df.STD)
df.Region = string.(df.Region)

histogram(df.Mean ,bins=10,label=nothing,xlabel="Control Error")

df_sort = sort!(df, [:Mean])