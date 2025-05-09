import pandas as pd
import numpy as np

# Let's make a simple dataframe

# First, let's create an index that uses pd.DateTime objects:
index = pd.Index(pd.date_range(start = "2025-5-1", end = "2025-5-2", freq = "min"), name = 'date')
print(index)
print("\n\n\n")


# Now, let's create data for five columns. I will make the dimension fit the index length:
df = pd.DataFrame(np.arange(5 * len(index)).reshape(len(index), -1),
        columns = ['a', 'b', 'c', 'd', 'e'],
        index = index)
print(df)
print("\n\n\n")

# simple selection of a dataframe returns a column as a pd.Series
print(df["a"])
print(f"Returns type: {str(type(df['a']))}")
print("\n\n\n")

# simple selection on the returned pd.Series is done by label:
print(df["a"]["2025-05-1 12:37:00"])
print("\n\n\n")

# In this slice, we take everything from one label to the next:
print(df["a"]["2025-05-1 00:00:00":"2025-05-1 12:00:00"])
print("\n\n\n")


# Let's select columns in the df, but have it returned as a df:
print(df[["b", "a"]])
print("\n\n\n")

# We can select BY LABEL all midnight times:
# First, we select columns in the order that we want them (returning a df)
# Next, we select by time BY LABEL in the returned df
result = df[["b", "a", "c"]].loc["2025", : ]
print(result)
print(type(result))
print("\n\n\n")

# We can also pull specific attributes from our df index:
print(df.index.year)
print(df.index.time)
print("\n\n\n")


# pd.Timestamp()
# Note: there is no pd.DateTime() constructor, as you'd think...
# In the following, let's create a new column called "time_delta"
# We will make it by taking the index (which contains pd.timestamps)
# Think of this column as a "time since data start" column...
df["time_delta"] = df.index - pd.Timestamp("2025-05-01 00:00:00")
df_rearranged = df[["time_delta", "a", "b", "c", "d", "e"]]
print(df_rearranged)

# Can we select by the time component (hour, minute, seconds)?
# We can do boolean slicing in this case. Recall the df.index.time from above.
# The pattern we wish to match is a pd.Timestamp().time:
midnight_rows = df.loc[df.index.time == pd.Timestamp("00:00:00").time()]
print(midnight_rows)





