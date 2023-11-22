#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as pd
import pandas as pd


# In[2]:


CCTV_Seoul = pd.read_csv('C:\\Users\\USER\\DataScience\\data\\CCTV_in_Seoul.csv', encoding = 'utf-8')
CCTV_Seoul.head()

# SyntaxError에러 발생                                                                               ^
#>> SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape

#해결 방법
# 폴더 경로 \ ->\\ 변경하여 사용


# In[3]:


CCTV_Seoul.columns


# In[4]:


CCTV_Seoul.drop(['Unnamed: 0'], axis=1, inplace = True)
CCTV_Seoul.columns


# In[5]:


CCTV_Seoul.rename(index = {'Nan':'합'},inplace=True)


# In[6]:


CCTV_Seoul.head()


# In[7]:


CCTV_Seoul.sort_values(by='총계', ascending=True).head()


# In[8]:


# 최근 5년간 CCTV 증가율
# TypeError: unsupported operand type(s) for /: 'str' and 'str'
# 데이터 값에 결측값이 있어 연산이 불가하다
# 결측값을 0으로 바꾸어 연산


# In[9]:


CCTV_Seoul[CCTV_Seoul['2014년 이전'].isnull()]


# In[10]:


CCTV_Seoul[CCTV_Seoul['2023년'].isnull()]


# In[11]:


CCTV_Seoul.drop([7], inplace = True)
CCTV_Seoul.drop([9], inplace = True)
CCTV_Seoul.drop([15], inplace = True)
CCTV_Seoul.drop([18], inplace = True)


# In[12]:


CCTV_Seoul[CCTV_Seoul['2023년'].isnull()]


# In[13]:


CCTV_Seoul[CCTV_Seoul['2014년 이전'].isnull()]


# In[14]:


CCTV_Seoul.head()


# In[15]:


CCTV_Seoul.sort_values(by='총계', ascending = False).head(10)


# In[16]:


CCTV_Seoul.sort_values(by='총계', ascending = True).head(10)


# In[17]:


#10개년 CCTV 증가율


# In[18]:


CCTV_Seoul['최근증가율'] = (CCTV_Seoul['2023년']+CCTV_Seoul['2022년']+CCTV_Seoul['2021년']+\
                       CCTV_Seoul['2020년']+ CCTV_Seoul['2019년']+CCTV_Seoul['2018년']+\
                       CCTV_Seoul['2017년']+CCTV_Seoul['2016년']+CCTV_Seoul['2015년']+\
                       CCTV_Seoul['2014년'])/CCTV_Seoul["2014년 이전"] * 100

CCTV_Seoul.sort_values(by='최근증가율',ascending=False).head(10)


# In[19]:


### 행정구 크기


# In[20]:


import pandas as pd
import numpy as np


# In[21]:


Region_size = pd.read_csv('C:\\Users\\USER\\DataScience\\data\\Region_size_in_Seoul.csv',header = 2, encoding='utf-8')
Region_size.head(10)


# In[22]:


# 또다른 열 삭제 방법 del Region_size['행정 (개)']
Region_size.drop('행정 (개)',axis=1,inplace=True)
Region_size.drop('법정 (개)',axis=1,inplace=True)
Region_size.drop('소계',axis=1,inplace=True)
Region_size.drop('소계.1',axis=1,inplace=True)


# In[23]:


Region_size.head(10)


# In[24]:


Region_size.sort_values(by  = '면적 (km²)', ascending = False).head(10)


# In[25]:


Region_size.rename(columns = {'자치구별(2)':'구분'},inplace=True)
Region_size.head(10)


# In[26]:


## 서울시 면적 대비 CCTV 설치 개수를 비교하고자 
## CCTV 데이터와 서울시 면적 데이터를 합치고 분석


# In[27]:


data_result = pd.merge(CCTV_Seoul, Region_size, on = '구분')

data_result.head(5)


# In[28]:


#필요 없는 열 삭제
del data_result['2023년']
del data_result['2022년']
del data_result['2021년']
del data_result['2020년']
del data_result['2019년']
del data_result['2018년']
del data_result['2017년']
del data_result['2016년']
del data_result['2015년']
del data_result['2014년']
del data_result['2014년 이전']


# In[29]:


data_result.head(5)


# In[30]:


data_result.set_index('구분',inplace = True)


# In[31]:


data_result.head()


# In[33]:


data_result.sort_values(by = '최근증가율', ascending=False).head()


# In[ ]:


## 두 데이터 변수간 상관계수(np.corcoef) 비교


# In[34]:


#구분-총계
data_result.sort_values(by = '총계', ascending=False).head()


# In[40]:


np.corrcoef(data_result['최근증가율'],data_result['총계'])

# 0.1이하의 값으로 적은 상관관계를 깆는다. 


# In[36]:


data_result.sort_values(by = '면적 (km²)', ascending=False).head()


# In[41]:


np.corrcoef(data_result['면적 (km²)'],data_result['총계'])

#array([[1.        , 0.40548212],
#       [0.40548212, 1.        ]])
## 0.405의 약한 상관관계를 갖는다. 


# In[42]:


data_result.sort_values(by='구성비 (%)',ascending=True).head()


# In[43]:


np.corrcoef(data_result['구성비 (%)'], data_result['총계'])


## 0.407의 약한 상관관계를 갖는다. 
# array([[1.        , 0.40799158],
#       [0.40799158, 1.        ]])


# In[44]:


##최근증가율-면적
np.corrcoef(data_result['최근증가율'], data_result['면적 (km²)'])


# In[45]:


##최근 증가율-구성비
np.corrcoef(data_result['최근증가율'], data_result['구성비 (%)'])


# In[46]:


## 면적-구성비
np.corrcoef(data_result['구성비 (%)'], data_result['면적 (km²)'])


# In[ ]:


### 결론 : 면적-구성비의 상관계수가 가장 높은 값을 가지므로, 더 높은 상관관계를 갖고 있다. 


# In[ ]:


#시각화


# In[74]:


import platform

from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Drarwin':
    rc('font',family='AppleGothic')
elif platform.system() == 'Windows':
    path="c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print("/Unknown system// Sorry")


# In[75]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[76]:


fp1 = np.polyfit(data_result['구성비 (%)'], data_result['면적 (km²)'],1)
fp1


# In[118]:


f1 = np.poly1d(fp1)
#가로축 :fx = np.linspace(시작점,끝점,간격)
fx = np.linspace(0,20,5)


# In[110]:


plt.figure(figsize=(6,6))
plt.scatter(data_result['구성비 (%)'], data_result['면적 (km²)'],s=30)
plt.plot(fx,f1(fx),ls='dashed', lw = 1.5,color='y')

plt.xlabel('구성비(%)')
plt.ylabel('면적(km)')

plt.grid()
plt.show()


# In[112]:


data_result['오차']= np.abs(data_result['면적 (km²)']-f1(data_result['구성비 (%)']))

df_sort = data_result.sort_values(by = '오차', ascending=False)
df_sort.head()


# In[122]:


f1 = np.poly1d(fp1)
#가로축 :fx = np.linspace(시작점,끝점,간격)
fx = np.linspace(0,10,1)


# In[124]:


plt.figure(figsize=(10,10))
plt.scatter(data_result['구성비 (%)'], data_result['면적 (km²)'], c=data_result['오차'],s=30)
plt.plot(fx,f1(fx),ls='dashed', lw = 1.5,color='y')

for n in range(10):
    plt.text(df_sort['구성비 (%)'][n]* 1.02,df_sort['면적 (km²)'][n]*0.98,
             df_sort.index[n],fontsize=10)
                                                                
plt.xlabel('구성비(%)')
plt.ylabel('면적(km)')

plt.colorbar()
plt.grid()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




