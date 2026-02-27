
Important to enable http communication when starting the instance


- load ngix
```
sudo apt update
sudo apt install nginx -y
```

- ngix config
```
sudo nano /etc/nginx/sites-available/fastapi
```

write 
```
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
^+O
Return
^+X

- Enable site
```
sudo ln -s /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled/
```

- Remove default config
```
sudo rm /etc/nginx/sites-enabled/default
```

- Test
```
sudo nginx -t
```
```
syntax is ok
test is successful
```

- Restart
```
sudo systemctl restart nginx
```

- Connect via
```
http://IP
```


- use https
```
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx
```
