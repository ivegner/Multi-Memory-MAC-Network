# Notes on performance

## Vanilla MAC
Epoch: 1; Loss: 0.79425; Acc: 0.59646: [30:38<00:00,  5.90it/s]
Avg Acc: 0.60788
Epoch: 2; Loss: 0.56873; Acc: 0.72860: [31:27<00:00,  5.79it/s]
Avg Acc: 0.76047
Epoch: 3; Loss: 0.30509; Acc: 0.86179: [31:08<00:00,  6.21it/s]
Avg Acc: 0.90701
Epoch: 4; Loss: 0.23143; Acc: 0.88911: [31:15<00:00,  6.57it/s]
Avg Acc: 0.92354
Epoch: 5; Loss: 0.28529; Acc: 0.90738: [31:37<00:00,  5.76it/s]
Avg Acc: 0.93522
Epoch: 6; Loss: 0.10179; Acc: 0.91826: [31:18<00:00,  5.82it/s]
Avg Acc: 0.94170
...
At 10 epochs, accuracy is supposed to be 95.75%
At 20, 96.5%

# 3 Memories (n=3) [times are on one GPU]
Epoch: 1; Loss: 0.91446; Acc: 0.51202: [1:07:42<00:00,  2.93it/s]
Avg Acc: 0.53016
Epoch: 2; Loss: 0.71245; Acc: 0.61467: [1:07:51<00:00,  3.05it/s]
Avg Acc: 0.63245
Epoch: 3; Loss: 0.41644; Acc: 0.84885: [1:08:15<00:00,  3.25it/s]
Avg Acc: 0.88836
Epoch: 4; Loss: 0.32514; Acc: 0.89475: [1:07:42<00:00,  3.35it/s]
Avg Acc: 0.92665
Epoch: 5; Loss: 0.11384; Acc: 0.91390: [1:07:45<00:00,  2.43it/s]
Avg Acc: 0.94342
Epoch: 6; Loss: 0.10911; Acc: 0.92774: [1:07:27<00:00,  3.04it/s]
Avg Acc: 0.94999
Epoch: 7; Loss: 0.03950; Acc: 0.93569: [1:07:30<00:00,  2.93it/s]
Avg Acc: 0.95545
Epoch: 8; Loss: 0.14394; Acc: 0.94439: [1:07:12<00:00,  3.39it/s]
Avg Acc: 0.95871
Epoch: 9; Loss: 0.06399; Acc: 0.94976: [1:08:19<00:00,  3.10it/s]
Avg Acc: 0.96146
Epoch: 10; Loss: 0.07438; Acc: 0.95375: [1:07:37<00:00,  2.86it/s]
Avg Acc: 0.96279
Epoch: 11; Loss: 0.24445; Acc: 0.95766: [1:08:20<00:00,  3.18it/s]
Avg Acc: 0.96477
Epoch: 12; Loss: 0.02926; Acc: 0.95835: [1:07:57<00:00,  3.02it/s]
Avg Acc: 0.96592
Epoch: 13; Loss: 0.05796; Acc: 0.96424: [1:08:02<00:00,  3.04it/s]
Avg Acc: 0.96622
Epoch: 14; Loss: 0.01756; Acc: 0.96504: [1:07:58<00:00,  3.00it/s]
Avg Acc: 0.96709
Epoch: 15; Loss: 0.00573; Acc: 0.96683: [1:08:11<00:00,  2.95it/s]
Avg Acc: 0.96742
Epoch: 16; Loss: 0.02573; Acc: 0.96864: [1:08:08<00:00,  3.33it/s]
Avg Acc: 0.96834

## 7 Memories (n=7)
Epoch: 1; Loss: 0.65650; Acc: 0.60635: [1:30:01<00:00,  2.13it/s]
Avg Acc: 0.61422
Epoch: 2; Loss: 0.41662; Acc: 0.72297: [1:30:25<00:00,  2.25it/s]
Avg Acc: 0.75226
Epoch: 3; Loss: 0.30118; Acc: 0.86245: [1:30:34<00:00,  2.32it/s]
Avg Acc: 0.90065
Epoch: 4; Loss: 0.19540; Acc: 0.89655: [1:30:26<00:00,  2.18it/s]
Avg Acc: 0.92178
Epoch: 5; Loss: 0.19991; Acc: 0.91657: [1:30:33<00:00,  2.09it/s]
Avg Acc: 0.94092
Epoch: 6; Loss: 0.17701; Acc: 0.93012: [1:30:38<00:00,  2.30it/s]
Avg Acc: 0.94876
Epoch: 7; Loss: 0.06979; Acc: 0.94454: [1:30:33<00:00,  2.14it/s]
Avg Acc: 0.95222
Epoch: 8; Loss: 0.07817; Acc: 0.94772: [1:30:25<00:00,  2.22it/s]
Avg Acc: 0.95426
Epoch: 9; Loss: 0.02039; Acc: 0.95195: [1:30:28<00:00,  2.19it/s]
Avg Acc: 0.95660
Epoch: 10; Loss: 0.12727; Acc: 0.95570: [1:31:24<00:00,  2.21it/s]
Avg Acc: 0.95769
