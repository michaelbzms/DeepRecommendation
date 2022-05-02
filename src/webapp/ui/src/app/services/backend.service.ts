import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Movie} from "../models/movies";

@Injectable({
  providedIn: 'root'
})
export class BackendService {

  constructor(private http: HttpClient) { }

  getMovies() {
    return this.http.get<any>('http://127.0.0.1:5000/movies');   // TODO: move url in environment
  }
}
