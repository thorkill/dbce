"""
Collection of constants for DBCE

Copyright (c) 2016 Rafal Lesniak

This software is licensed as described in the file LICENSE.
"""

DBCE_MARKUP_ANALYSIS_TAG = "dbce"
DBCE_MARKUP_PRESENTATION_TAG = "span"

DBCE_CONTENT_NAME = "other-Uc9heeK0"
DBCE_DIFF_CUR = "cur-Gah5aipa"
DBCE_DIFF_DOWN = "down-aeMaix1i"
DBCE_DIFF_UP = "up-Zaachae8"
DBCE_DIFF_NEXT = "next-ooTheiW0"
DBCE_DIFF_PREV = "prev-AhCho7aa"

DBCE_CLASS_UNKNWON = "dbce-class-unk"
DBCE_CLASS_BOILER_PLATE = "dbce-class-bp"
DBCE_CLASS_CONTENT = "dbce-class-content"
DBCE_CLASS_INVALID = "dbce-class-invalid"

DBCE_BORDER_TAGS = ['body', 'div', 'footer', 'aside', 'header']

# (CUR, UP, DOWN, PREV, NEXT).
DBCE_BITS_TO_CLASS = {
    # cur + up
    (1, 1, 0, 0, 0) : {
        (1, 0, 0, 0, 0): DBCE_CLASS_CONTENT,
        (1, 1, 0, 0, 0): DBCE_CLASS_BOILER_PLATE,
    },
    # cur + up + down
    (1, 1, 1, 0, 0) : {
        (1, 0, 0, 0, 0): DBCE_CLASS_CONTENT,
        (1, 0, 1, 0, 0): DBCE_CLASS_BOILER_PLATE,
        (1, 1, 0, 0, 0): DBCE_CLASS_BOILER_PLATE,
        (1, 1, 1, 0, 0): DBCE_CLASS_BOILER_PLATE,
    },
    # cur + up + next
    (1, 1, 0, 0, 1) : {
        (1, 0, 0, 0, 0): DBCE_CLASS_BOILER_PLATE,
        (1, 0, 0, 0, 1): DBCE_CLASS_CONTENT,
        (1, 1, 0, 0, 0): DBCE_CLASS_BOILER_PLATE,
        (1, 1, 0, 0, 1): DBCE_CLASS_BOILER_PLATE,
    },
    # cur + up + prev
    (1, 1, 0, 1, 0) : {
        (1, 0, 0, 0, 0): DBCE_CLASS_BOILER_PLATE,
        (1, 0, 0, 1, 0): DBCE_CLASS_CONTENT,
        (1, 1, 0, 0, 0): DBCE_CLASS_BOILER_PLATE,
        (1, 1, 0, 1, 0): DBCE_CLASS_BOILER_PLATE,
    },
    # cur + up + prev + next
    (1, 1, 0, 1, 1) : {
        (1, 0, 0, 0, 0): DBCE_CLASS_BOILER_PLATE,
        (1, 0, 0, 0, 1): DBCE_CLASS_BOILER_PLATE,
        (1, 0, 0, 1, 0): DBCE_CLASS_BOILER_PLATE,
        (1, 0, 0, 1, 1): DBCE_CLASS_CONTENT,
        (1, 1, 0, 0, 0): DBCE_CLASS_BOILER_PLATE,
        (1, 1, 0, 0, 1): DBCE_CLASS_BOILER_PLATE,
        (1, 1, 0, 1, 0): DBCE_CLASS_BOILER_PLATE,
        (1, 1, 0, 1, 1): DBCE_CLASS_BOILER_PLATE,
    },
    # cur + up + down + next
    (1, 1, 1, 0, 1) : {
        (1, 0, 0, 0, 0): DBCE_CLASS_INVALID,
        (1, 0, 0, 0, 1): DBCE_CLASS_CONTENT,
        (1, 0, 1, 0, 0): DBCE_CLASS_INVALID,
        (1, 0, 1, 0, 1): DBCE_CLASS_INVALID,
        (1, 1, 0, 0, 0): DBCE_CLASS_INVALID,
        (1, 1, 0, 0, 1): DBCE_CLASS_INVALID,
        (1, 1, 1, 0, 0): DBCE_CLASS_INVALID,
        (1, 1, 1, 0, 1): DBCE_CLASS_BOILER_PLATE,
        },
    # cur + up + down + prev
    (1, 1, 1, 1, 0) : {
        (1, 0, 0, 0, 0): DBCE_CLASS_INVALID,
        (1, 0, 0, 1, 0): DBCE_CLASS_CONTENT,
        (1, 0, 1, 0, 0): DBCE_CLASS_INVALID,
        (1, 0, 1, 1, 0): DBCE_CLASS_INVALID,
        (1, 1, 0, 0, 0): DBCE_CLASS_INVALID,
        (1, 1, 0, 1, 0): DBCE_CLASS_INVALID,
        (1, 1, 1, 0, 0): DBCE_CLASS_INVALID,
        (1, 1, 1, 1, 0): DBCE_CLASS_BOILER_PLATE,
        },
    # cur + up + down + prev + next
    (1, 1, 1, 1, 1) : {
        (1, 0, 0, 0, 0): DBCE_CLASS_INVALID,
        (1, 0, 0, 0, 1): DBCE_CLASS_INVALID,
        (1, 0, 0, 1, 0): DBCE_CLASS_INVALID,
        (1, 0, 0, 1, 1): DBCE_CLASS_CONTENT,
        (1, 0, 1, 0, 0): DBCE_CLASS_INVALID,
        (1, 0, 1, 0, 1): DBCE_CLASS_INVALID,
        (1, 0, 1, 1, 0): DBCE_CLASS_INVALID,
        (1, 0, 1, 1, 1): DBCE_CLASS_INVALID,
        (1, 1, 0, 0, 0): DBCE_CLASS_INVALID,
        (1, 1, 0, 0, 1): DBCE_CLASS_INVALID,
        (1, 1, 0, 1, 0): DBCE_CLASS_INVALID,
        (1, 1, 0, 1, 1): DBCE_CLASS_INVALID,
        (1, 1, 1, 0, 0): DBCE_CLASS_INVALID,
        (1, 1, 1, 0, 1): DBCE_CLASS_INVALID,
        (1, 1, 1, 1, 0): DBCE_CLASS_INVALID,
        (1, 1, 1, 1, 1): DBCE_CLASS_BOILER_PLATE
    }
}

# https://developer.mozilla.org/en-US/docs/Web/HTML/Element
DBCE_VALID_TAGS = ["a", "abbr", "acronym", "address", "applet", "area", "article", "aside", "audio",
                   "b", "base", "basefont", "bdi", "bdo", "big", "blockquote", "body", "br", "button",
                   "canvas", "caption", "center", "cite", "code", "col", "colgroup",
                   "datalist", "dd", "del", "details", "dfn", "dialog", "dir", "div", "dl", "dt",
                   "em", "embed",
                   "fieldset", "figcaption", "figure", "font", "footer", "form", "frame", "frameset",
                   "h1", "h2", "h3", "h4", "h5", "head", "header",
                   "hr", "html",
                   "i", "iframe", "img", "input", "ins",
                   "kbd", "keygen",
                   "label", "legend", "li", "link",
                   "main", "map", "mark", "math", "menu", "menuitem", "meta", "meter",
                   "nav", "noframes", "noscript",
                   "object", "ol", "optgroup", "option", "output",
                   "p", "param", "picture", "pre", "progress",
                   "q",
                   "rp", "rt", "ruby",
                   "s", "samp", "script", "section", "select", "small", "source", "span", "strike",
                   "strong", "style", "sub", "summary", "sup", "svg",
                   "table", "tbody", "td", "textarea", "tfoot", "th", "thead", "time", "title",
                   "tr", "track", "tt",
                   "u", "ul",
                   "var", "video",
                   "wbr"]


DBCE_STYLE = """

.{} {{ border:2px; border-style:solid; border-color:#FF0000; padding: 1px; background-color: #ff8000;}}
.{} {{ border:2px; border-style:solid; border-color:#248f24; padding: 1px; background-color: #70db70;}}
.{} {{ border:2px; border-style:solid; border-color:#999900; padding: 1px; background-color: #ffff66;}}
.{} {{ border:2px; border-style:solid; border-color:#3d5c5c; padding: 1px; background-color: #669999;}}

.{}a {{ border:2px; border-style:solid; border-color:#FFcc00; padding: 1px; background-color: #ffcccc;}}
.{}a {{ border:2px; border-style:dotted; border-color:#ccFF00; padding: 1px; background-color: #66ff66;}}
.{}a {{ border:2px; border-style:dotted; border-color:#178819; padding: 1px; background-color: #66aa11;}}
.{}a {{ border:2px; border-style:dotted; border-color:#cc00aa; padding: 1px; background-color: #11aa66;}}
.{}a {{ border:2px; border-style:dotted; border-color:#212413; padding: 1px; background-color: #ccddaa;}}
.{}a {{ border:2px; border-style:dotted; border-color:#4567ad; padding: 1px; background-color: #981257;}}
            """.format(DBCE_CLASS_BOILER_PLATE,
                       DBCE_CLASS_CONTENT,
                       DBCE_CLASS_INVALID,
                       DBCE_CLASS_UNKNWON,
                       DBCE_DIFF_CUR, DBCE_CONTENT_NAME, DBCE_DIFF_UP,
                       DBCE_DIFF_DOWN, DBCE_DIFF_NEXT, DBCE_DIFF_PREV)
