import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HttpClientModule } from '@angular/common/http';
import {TableModule} from "primeng/table";
import {ButtonModule} from "primeng/button";
import {RatingModule} from "primeng/rating";
import { FormsModule } from '@angular/forms';
import {CardModule} from "primeng/card";

@NgModule({
  declarations: [
    AppComponent
  ],
    imports: [
        BrowserModule,
        AppRoutingModule,
        HttpClientModule,
        TableModule,
        ButtonModule,
        RatingModule,
        FormsModule,
        CardModule
    ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
