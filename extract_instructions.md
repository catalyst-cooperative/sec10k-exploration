## Overview
SEC 10-k filings have an exhibit 21 attachment where companies list any subsidiary companies they
own either in whole or in part. You will be extracting this information about subsidiary companies
and returning structured JSON. Below are a number of rules to follow to do this extraction as
desired.

The exhibit 21 is HTML, but it is not consistently structured. Sometimes there is just a list of
subsidiary companies and sometimes it may have more structure like a table. It will also sometimes
have information like the percent of a company that is owned by the original filer or the location of
the subsidiary company. There are also cases where it is displayed as a hierarchy of ownership. This
might not be defined explicitly, but rather through the structure of the HTML. For example it may use
indentation to show that a company is actually a subsidiary of another subsidiary company.

## Rules
1. You will be returning a list of subsidiary companies as structured JSON data
2. There are several fields you are looking for including:
  - `company_name`: Name of subsidiary company. This is the only required field. If it can't be determined definitively, then do not add an entry at all.
  - `location_of_incorporation`: Location of the subsidiary. If a location does not exist, or can't be clearly determined then this should be Null.
  - `fraction_owned_by_parent`: Fraction of the subsidiary company that is owned by the filer. If a ownership fraction does not exist, or can't be clearly determined then this should be Null.
  - `immediate_parent`: The immediate parent of the subsidiary company if ownership is displayed as a hierarchical structure. If no hierarchy exists, or can not be clearly determined this should be Null.
3. Give names **exactly** as they are found in the filing. Do not attempt to clean this up at all.
4. 
