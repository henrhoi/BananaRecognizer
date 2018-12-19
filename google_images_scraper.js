// Add jQuery-script to head of document, through the JavaScript console
const script = document.createElement('script');
script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(script);

// Collect the URLs
const urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });

// Write the URLs to file.
const urlsToSave = urls.toArray().join('\n');
const download_button = document.createElement('a');
download_button.href = 'data:attachment/text,' + encodeURI(urlsToSave);
download_button.target = '_blank';
download_button.download = 'urls.txt';
download_button.click();

