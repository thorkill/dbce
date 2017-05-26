"""
dbcediff - HTML annotator/differ based on lxml.html.diff

Copyright (c) 2016 Rafal Lesniak. All rights reserved.
Copyright (c) 2004 Ian Bicking. All rights reserved.

"""

import difflib
import re
import pdb

#from html import escape as html_escape

from lxml import etree
from lxml.html import fragment_fromstring

from xml.sax.saxutils import escape

#__all__ = ['html_annotate', 'htmldiff']

#basestring = str

SPLIT_WORDS_RE = re.compile(r'\S+(?:\s+|$)', re.U)
START_WHITESPACE_RE = re.compile(r'^[ \t\n\r]')

############################################################
## Annotation
############################################################

HTML_ESCAPE_TABLE = {
    '"': "&quot;",
    "'": "&apos;"
}

def html_escape(text):
    return escape(text, HTML_ESCAPE_TABLE)

def default_markup(text, version):
    _version = " ".join(version)
    return '<dbce class="%s">%s</dbce>' % (
        html_escape(str(_version)), text)

def html_annotate(doclist, markup=default_markup, compress=False, window_size=0):
    """
    doclist should be ordered from oldest to newest, like::

        >>> version1 = 'Hello World'
        >>> version2 = 'Goodbye World'
        >>> print(html_annotate([(version1, 'version 1'),
        ...                      (version2, 'version 2')]))
        <span title="version 2">Goodbye</span> <span title="version 1">World</span>

    The documents must be *fragments* (str/UTF8 or unicode), not
    complete documents

    The markup argument is a function to markup the spans of words.
    This function is called like markup('Hello', 'version 2'), and
    returns HTML.  The first argument is text and never includes any
    markup.  The default uses a span with a title:

        >>> print(default_markup('Some Text', 'by Joe'))
        <span title="by Joe">Some Text</span>
    """
    # The basic strategy we have is to split the documents up into
    # logical tokens (which are words with attached markup).  We then
    # do diffs of each of the versions to track when a token first
    # appeared in the document; the annotation attached to the token
    # is the version where it first appeared.
    tokenlist = [tokenize_annotated(doc, version)
                 for doc, version in doclist]
    cur_tokens = tokenlist[0]
    for tokens in tokenlist[1:]:
        # cur_tokens == old_version
        # tokens == new_version
        # sequence matching will be performed
        html_annotate_merge_annotations(cur_tokens, tokens)
        cur_tokens = tokens

    # After we've tracked all the tokens, we can combine spans of text
    # that are adjacent and have the same annotation
    if compress:
        cur_tokens = compress_tokens(cur_tokens, window_size=window_size)
    # And finally add markup
    result = markup_serialize_tokens(cur_tokens, markup)
    return ''.join(result).strip()

def tokenize_annotated(doc, annotation):
    """Tokenize a document and add an annotation attribute to each token
    """
    tokens = tokenize(doc, include_hrefs=False)
    for tok in tokens:
        tok.annotation = [annotation]
    return tokens

def html_annotate_merge_annotations(tokens_old, tokens_new):
    """Merge the annotations from tokens_old into tokens_new, when the
    tokens in the new document already existed in the old document.
    """
    s = InsensitiveSequenceMatcher(a=tokens_old, b=tokens_new)
    commands = s.get_opcodes()

    for command, i1, i2, j1, j2 in commands:
        if command == 'equal':
            eq_old = tokens_old[i1:i2]
            eq_new = tokens_new[j1:j2]
            copy_annotations(eq_old, eq_new)

def copy_annotations(src, dest):
    """
    Copy annotations from the tokens listed in src to the tokens in dest
    """
    assert len(src) == len(dest)
    for src_tok, dest_tok in zip(src, dest):
        for s_t_ann in src_tok.annotation:
            if s_t_ann not in dest_tok.annotation:
                dest_tok.annotation.append(s_t_ann)

def compress_tokens(tokens, window_size=0):
    """
    Combine adjacent tokens when there is no HTML between the tokens,
    and they share an annotation
    """

    result = [tokens[0]]
    result[-1].annotation.sort()
    compress_count = 1
    for tok in tokens[1:]:
        tok.annotation.sort()

        _compress = (window_size > 0 and compress_count < window_size) or window_size == 0

        if not (result[-1].post_tags or tok.pre_tags) \
           and result[-1].annotation == tok.annotation \
           and _compress:
            compress_merge_back(result, tok)
            compress_count += 1
        else:
            result.append(tok)
            compress_count = 1
    return result

def compress_merge_back(tokens, tok):
    """ Merge tok into the last element of tokens (modifying the list of
    tokens in-place).  """

    last = tokens[-1]
    if not isinstance(last, Token) or not isinstance(tok, Token):
        tokens.append(tok)
    else:
        text = str(last)
        if last.trailing_whitespace:
            text += last.trailing_whitespace
        text += tok
        merged = Token(text,
                       pre_tags=last.pre_tags,
                       post_tags=tok.post_tags,
                       trailing_whitespace=tok.trailing_whitespace)
        merged.annotation = last.annotation
        tokens[-1] = merged

def markup_serialize_tokens(tokens, markup_func):
    """
    Serialize the list of tokens into a list of text chunks, calling
    markup_func around text to add annotations.
    """
    for token in tokens:
        for pre in token.pre_tags:
            yield pre
        html = token.html()
        html = markup_func(html, token.annotation)
        if token.trailing_whitespace:
            html += token.trailing_whitespace
        yield html
        for post in token.post_tags:
            yield post

class NoDeletes(Exception):
    """ Raised when the document no longer contains any pending deletes
    (DEL_START/DEL_END) """

class Token(str):
    """ Represents a diffable token, generally a word that is displayed to
    the user.  Opening tags are attached to this token when they are
    adjacent (pre_tags) and closing tags that follow the word
    (post_tags).  Some exceptions occur when there are empty tags
    adjacent to a word, so there may be close tags in pre_tags, or
    open tags in post_tags.

    We also keep track of whether the word was originally followed by
    whitespace, even though we do not want to treat the word as
    equivalent to a similar word that does not have a trailing
    space."""

    # When this is true, the token will be eliminated from the
    # displayed diff if no change has occurred:
    hide_when_equal = False

    def __new__(cls, text, pre_tags=None, post_tags=None, trailing_whitespace=""):
        obj = str.__new__(cls, text)

        if pre_tags is not None:
            obj.pre_tags = pre_tags
        else:
            obj.pre_tags = []

        if post_tags is not None:
            obj.post_tags = post_tags
        else:
            obj.post_tags = []

        obj.trailing_whitespace = trailing_whitespace
        obj.annotation = []

        return obj

    def __repr__(self):
        return 'Token(%s, %r, %r, %r)' % (str.__repr__(self),
                                          self.pre_tags,
                                          self.post_tags,
                                          self.trailing_whitespace)

    def html(self):
        return str(self)

class TagToken(Token):

    """ Represents a token that is actually a tag.  Currently this is just
    the <img> tag, which takes up visible space just like a word but
    is only represented in a document by a tag.  """

    def __new__(cls, tag, data, html_repr, pre_tags=None,
                post_tags=None, trailing_whitespace=""):
        obj = Token.__new__(cls, "%s: %s" % (type, data),
                            pre_tags=pre_tags,
                            post_tags=post_tags,
                            trailing_whitespace=trailing_whitespace)
        obj.tag = tag
        obj.data = data
        obj.html_repr = html_repr
        return obj

    def __repr__(self):
        return 'TagToken(%s, %s, html_repr=%s, pre_tags=%r, post_tags=%r, trailing_whitespace=%r)' % (
            self.tag,
            self.data,
            self.html_repr,
            self.pre_tags,
            self.post_tags,
            self.trailing_whitespace)

    def html(self):
        return self.html_repr

class HrefToken(Token):

    """ Represents the href in an anchor tag.  Unlike other words, we only
    show the href when it changes.  """

    hide_when_equal = True

    def html(self):
        return ' Link: %s' % self

def tokenize(html, include_hrefs=True):
    """
    Parse the given HTML and returns token objects (words with attached tags).

    This parses only the content of a page; anything in the head is
    ignored, and the <head> and <body> elements are themselves
    optional.  The content is then parsed by lxml, which ensures the
    validity of the resulting parsed document (though lxml may make
    incorrect guesses when the markup is particular bad).

    <ins> and <del> tags are also eliminated from the document, as
    that gets confusing.

    If include_hrefs is true, then the href attribute of <a> tags is
    included as a special kind of diffable token."""
    if etree.iselement(html):
        body_el = html
    else:
        body_el = parse_html(html, cleanup=True)
    # Then we split the document into text chunks for each tag, word, and end tag:
    chunks = flatten_el(body_el, skip_tag=True, include_hrefs=include_hrefs)
    # Finally re-joining them into token objects:
    return fixup_chunks(chunks)

def parse_html(html, cleanup=True):
    """
    Parses an HTML fragment, returning an lxml element.  Note that the HTML will be
    wrapped in a <div> tag that was not in the original document.

    If cleanup is true, make sure there's no <head> or <body>, and get
    rid of any <ins> and <del> tags.
    """
    if cleanup:
        # This removes any extra markup or structure like <head>:
        html = cleanup_html(html)
    return fragment_fromstring(html, create_parent=True)

RE_BODY = re.compile(r'<body.*?>', re.I|re.S)
RE_END_BODY = re.compile(r'</body.*?>', re.I|re.S)
RE_INS_DEL = re.compile(r'</?(ins|del).*?>', re.I|re.S)
RE_END_WHITESPACE = re.compile(r'[ \t\n\r]$')

def cleanup_html(html):
    """ This 'cleans' the HTML, meaning that any page structure is removed
    (only the contents of <body> are used, if there is any <body).
    Also <ins> and <del> tags are removed.  """
    match = RE_BODY.search(html)
    if match:
        html = html[match.end():]
    match = RE_END_BODY.search(html)
    if match:
        html = html[:match.start()]
    html = RE_INS_DEL.sub('', html)
    return html

def split_trailing_whitespace(word):
    """
    This function takes a word, such as 'test\n\n' and returns ('test','\n\n')
    """
    stripped_length = len(word.rstrip())
    return word[0:stripped_length], word[stripped_length:]

def fixup_chunks(chunks):
    """
    This function takes a list of chunks and produces a list of tokens.
    """
    tag_accum = []
    result = []

    for chunk in chunks:
        if isinstance(chunk, tuple):
            if chunk[0] == 'img':
                src = chunk[1]
                tag, trailing_whitespace = split_trailing_whitespace(chunk[2])
                cur_token = TagToken('img',
                                     src,
                                     html_repr=tag,
                                     pre_tags=tag_accum,
                                     trailing_whitespace=trailing_whitespace)
                tag_accum = []
                result.append(cur_token)
            elif chunk[0] == 'href':
                href = chunk[1]
                cur_href = HrefToken(href, pre_tags=tag_accum, trailing_whitespace=" ")
                tag_accum = []
                result.append(cur_href)
            continue


        if is_word(chunk):
            chunk, trailing_whitespace = split_trailing_whitespace(chunk)
            cur_word = Token(chunk, pre_tags=tag_accum, trailing_whitespace=trailing_whitespace)
            tag_accum = []
            result.append(cur_word)

        elif is_start_tag(chunk):
            tag_accum.append(chunk)

        elif is_end_tag(chunk):
            if len(tag_accum) == 0:
                result[-1].post_tags.extend([chunk])
            elif len(tag_accum):
                tag_accum.append(chunk)
            else:
                assert cur_word, (
                    "Weird state, cur_word=%r, result=%r, chunks=%r of %r"
                    % (cur_word, result, chunk, chunks))
                cur_word.post_tags.append(chunk)
        else:
            assert 0

    if not result:
        return [Token('', pre_tags=tag_accum)]
    else:
        result[-1].post_tags.extend(tag_accum)

    return result


# All the tags in HTML that don't require end tags:
EMPTY_TAGS = (
    'param', 'img', 'area', 'br', 'basefont', 'input',
    'base', 'meta', 'link', 'col')

BLOCK_LEVEL_TAGS = (
    'address',
    'blockquote',
    'center',
    'dir',
    'div',
    'dl',
    'fieldset',
    'form',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'hr',
    'isindex',
    'menu',
    'noframes',
    'noscript',
    'ol',
    'p',
    'pre',
    'table',
    'ul',
    )

BLOCK_LEVEL_CONTAINER_TAGS = (
    'dd',
    'dt',
    'frameset',
    'li',
    'tbody',
    'td',
    'tfoot',
    'th',
    'thead',
    'tr',
    )

def flatten_el(el, include_hrefs, skip_tag=False):
    """ Takes an lxml element el, and generates all the text chunks for
    that tag.  Each start tag is a chunk, each word is a chunk, and each
    end tag is a chunk.

    If skip_tag is true, then the outermost container tag is
    not returned (just its contents)."""
    if not skip_tag:
        if el.tag == 'img':
            yield ('img', el.get('src'), start_tag(el))
        else:
            yield start_tag(el)
    if el.tag in EMPTY_TAGS and not el.text and not len(el) and not el.tail:
        return

    # start_words
    for word in map(html_escape, split_words(el.text)):
        yield word
    for child in el:
        for item in flatten_el(child, include_hrefs=include_hrefs):
            yield item
    if el.tag == 'a' and el.get('href') and include_hrefs:
        yield ('href', el.get('href'))
    if not skip_tag:
        yield end_tag(el)
        end_words = split_words(el.tail)
        for word in end_words:
            yield html_escape(word)

def split_words(text):
    """ Splits some text into words. Includes trailing whitespace
    on each word when appropriate.  """
    if not text or not text.strip():
        return []

    words = SPLIT_WORDS_RE.findall(text)
    return words

def start_tag(el):
    """
    The text representation of the start tag for a tag.
    """
    return '<%s%s>' % (
        el.tag, ''.join(map(lambda x: ' %s="%s"' % (x[0], html_escape(x[1])),
                            el.attrib.items())))

def end_tag(el):
    """ The text representation of an end tag for a tag.  Includes
    trailing whitespace when appropriate.  """
    if el.tail and START_WHITESPACE_RE.search(el.tail):
        extra = ' '
    else:
        extra = ''
    return '</%s>%s' % (el.tag, extra)

def is_word(tok):
    """ Returns True if token does not start with '<'"""
    return not tok.startswith('<')

def is_end_tag(tok):
    """ Returns True if token does starts with '</' """
    return tok.startswith('</')

def is_start_tag(tok):
    """ Returns True if token does starts with '<' and
    not with '</'
    """
    return tok.startswith('<') and not tok.startswith('</')

class InsensitiveSequenceMatcher(difflib.SequenceMatcher):
    """
    Acts like SequenceMatcher, but tries not to find very small equal
    blocks amidst large spans of changes
    """

    threshold = 2

    def get_matching_blocks(self):
        size = min(len(self.a), len(self.b))
        threshold = min(self.threshold, size / 4)
        actual = difflib.SequenceMatcher.get_matching_blocks(self)
        return [item for item in actual
                if item[2] > threshold
                or not item[2]]
