import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root',
})
export class PredictService {
  _url = 'http://localhost:5000/predict';
  constructor(private _http: HttpClient) {}

  predict(short: String) {
    return this._http.post<any>(this._url, short);
  }
}
