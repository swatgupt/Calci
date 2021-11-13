import streamlit as st
import numpy as np 
def calculate_emi(p,n,r):

	formula3_num = (1+(r/100))**n
	formula3_den = ((1+(r/100))**n) -1
	final_formula = p*(r/100)*(formula3_num/formula3_den)
	return final_formula

def calculate_outstanding_balance(p,n,r,m):
	term1 = (1 + (r/100))**n
	term2 = (1+ (r/100))**m
	den = (((1 + (r/100))**n) - 1)
	final_formula1 = (p*term1-term2) / den
	return final_formula1

st.title("EMI CALCULATOR")
principal = st.sidebar.slider("Principal",1000,1000000)
tenure = st.sidebar.slider("Tenure",1,30)
roi = st.sidebar.slider("Rate Of Interst",1,15)
m = st.sidebar.slider("Months",1,tenure*12)
r = roi/12
n = tenure * 12
if st.button("CALCULATE EMI"):
	emi = calculate_emi(principal,n,r)
	st.write("EMI:",round(emi,3))
if st.button("CALCULATE OUTSTANDING LOAN BALANCE"):
	loan_balance = calculate_outstanding_balance(principal,n,r,m)
	st.write("OUTSTANDING LOAN BALANCE:",round(loan_balance,3))