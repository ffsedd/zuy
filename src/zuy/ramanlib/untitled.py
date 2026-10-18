from bs4 import BeautifulSoup

html_content = """
<div class="field">
    <a data-target="dlmodal" onclick="updateFormAction('/index.php?lib=sop&id=PR1_E_785_kikirpa&ds=baseline corrected&dl=4zf2yf');" class="button is-fullwidth is-dark modal-button">
        <span class="icon"><i class="fa fa-download"></i></span>
        <span>ASCII (.csv)</span>
    </a>
</div>
<div class="field">
    <a data-target="dlmodal" onclick="updateFormAction('/index.php?lib=sop&id=PR1_E_785_kikirpa&ds=baseline corrected&dl=ar83kn');" class="button is-fullwidth is-dark modal-button">
        <span class="icon"><i class="fa fa-download"></i></span>
        <span>JCAMPDX (.dx)</span>
    </a>
</div>
"""

# Parse HTML
soup = BeautifulSoup(html_content, "html.parser")

# Find all divs with class "field"
field_divs = soup.find_all("div", class_="field")

# Extract onclick URL from divs containing the specified span
onclick_urls = []
for div in field_divs:
    anchor_tag = div.find("a")
    span_tag = div.find("span", text="JCAMPDX (.dx)")
    if anchor_tag and span_tag:
        onclick_attr = anchor_tag.get("onclick")
        if onclick_attr:
            url_start_index = onclick_attr.find("'") + 1
            url_end_index = onclick_attr.rfind("'")
            onclick_url = onclick_attr[url_start_index:url_end_index]
            onclick_urls.append(onclick_url)

print("Extracted onclick URLs:")
for url in onclick_urls:
    print(url)
