
var forceRedraw = function(element){
    var disp = element.style.display;
    element.style.display = 'none';
    var trick = element.offsetHeight;
    element.style.display = disp;
};

function unwrap(el) {
    // get the element's parent node
    var parent = el.parentNode;
    // move all children out of the element
    while (el.firstChild) parent.insertBefore(el.firstChild, el);
    // remove the empty element
    parent.removeChild(el);
//    forceRedraw(parent);
    return false;
}

function doSomething(el) {
    console.log(el);
    // nodeType 3 - text node
    if (el.nextSibling.nodeType == 3) {
        alert("will merge el withg el.nextSibling");
        el.appendChild(el.nextSibling);
        return true;

    }
    else if (el.parentElement.nodeType == 1 && el.parentElement.parentElement) {
        // nodeType 1 - a href
        var para = document.createElement("dbce");
        para.className = el.className;
        el.parentElement.parentElement.appendChild(para);
    }

    return false;
}

function toggleBP() {

    var names = ["dbce-marker","dbce-marker-child", "dbce-marker-element"];
    //var skipTags = ["SPAN", "DIV", "P"]
    var skipTags = [];
    for (var n in names) {
        var alldbces = document.getElementsByClassName(names[n]);
        for (var i=alldbces.length; i > 0; i--) {
            el = alldbces[i-1];
            if (((el.classList.contains('dbce-class-bp') == true)) &&
                ((skipTags.indexOf(el.tagName) == -1)))

            {
                if (el.style.visibility == 'hidden') {
                    el.style.visibility = "visible";
                } else {
                    el.style.visibility = "hidden";
                }
            } else {
                continue;
            }
        }
    }
}


function getElementsByAttribute(attribute, context) {
    var nodeList = (context || document).getElementsByTagName('*');
    var nodeArray = [];
    var iterator = 0;
    var node = null;

    while (node = nodeList[iterator++]) {
        if (node.hasAttribute(attribute)) nodeArray.push(node);
    }

    return nodeArray;
}

function toggleClassType() {
    var nodeList = document.querySelectorAll('[dbce-class]');

    var iterator = 0;
    var node = null;

    while (node = nodeList[iterator++]) {
        if (node.attributes['dbce-class'].value == 'dbce-class-bp') {
            node.remove();
        }
    }

    return;
}
