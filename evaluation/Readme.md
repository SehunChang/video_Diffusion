run_eval.py에서 나머지 evaluation 파일 한번에 돌아가도록 한 후 out_csv에 저장되도록 되어있음.

ex.) python3 or python run_eval.py --fake_dir {fake directory} --real_dir {} --out_csv {설정 안할 시 results.csv}

aesthetic 을 돌리기 위해서는 코드 내 깃허브에서 pth 파일 다운로드 (sac+logos+ava1-l14-linearMSE.pth) 후 저장해둬야 함.
run_eval.py에서 ev_aes에 해당
