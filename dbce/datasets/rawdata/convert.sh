#!/bin/sh

warcfilter -U www.heise.de thkr_holy-20160921224209.warc.gz thkr_holy-20160921224542.warc.gz thkr_holy-20160926150105.warc.gz thkr_holy-20161022030417.warc.gz > dbce-heise.warc
warcfilter -U www.spiegel.de thkr_holy-20160921224209.warc.gz thkr_holy-20160921224542.warc.gz thkr_holy-20160926150105.warc.gz thkr_holy-20161022030417.warc.gz > dbce-spiegel.warc
warcfilter -U www.wired.com thkr_holy-20160921224209.warc.gz thkr_holy-20160921224542.warc.gz thkr_holy-20160926150105.warc.gz thkr_holy-20161022030417.warc.gz > dbce-wired.warc
warcfilter -U www.chefkoch.de thkr_holy-20160921224209.warc.gz thkr_holy-20160921224542.warc.gz thkr_holy-20160926150105.warc.gz thkr_holy-20161022030417.warc.gz > dbce-chefkoch.warc

warc2warc -Z dbce-heise.warc > dbce-heise.warc.gz
warc2warc -Z dbce-spiegel.warc > dbce-spiegel.warc.gz
warc2warc -Z dbce-wired.warc > dbce-wired.warc.gz
warc2warc -Z dbce-chefkoch.warc > dbce-chefkoch.warc.gz

cdx-indexer dbce-heise.warc.gz > dbce-heise.cdx
cdx-indexer dbce-spiegel.warc.gz > dbce-spiegel.cdx
cdx-indexer dbce-wired.warc.gz > dbce-wired.cdx
cdx-indexer dbce-chefkoch.warc.gz > dbce-chefkoch.cdx

gzip -v dbce-*.cdx
